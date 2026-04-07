import math
from contextlib import contextmanager
import logging
from time import perf_counter

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import POSTGRES_DSN

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            POSTGRES_DSN,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def get_connection():
    engine = get_engine()
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        logger.exception("DB transaction failed, rolled back")
        raise
    finally:
        conn.close()


def fetch_property_analytics(
    tipo_transaccion: str | None = None,
    segmento: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Carga datos desde Postgres filtrando opcionalmente por tipo_transaccion y segmento.
    Solo trae las columnas necesarias para entrenamiento (sin mlsid, cluster_zona,
    subtipo_original, categoria_propiedad, status, transaction_type).
    """
    base_query = """
        SELECT
            id_propiedad,
            tipo_propiedad,
            estado_propiedad,
            segmento,
            tipo_transaccion,
            cluster_zona,
            latitude,
            longitude,
            ciudad,
            pais,
            m2_construidos,
            m2_terreno,
            dormitorios,
            banos,
            estacionamientos,
            antiguedad,
            precio_publicacion,
            precio_venta,
            precio_alquiler_mes,
            precio_m2,
            tiempo_en_mercado,
            numero_reducciones,
            diferencia_vs_promedio_zona,
            ratio_activas_vendidas_zona,
            mes_publicacion,
            anio_publicacion,
            fecha_venta,
            fecha_carga,
            fecha_actualizacion
        FROM public.property_analytics
        WHERE 1=1
    """

    params: dict = {}

    if tipo_transaccion is not None:
        base_query += " AND tipo_transaccion = :tipo_transaccion"
        params["tipo_transaccion"] = tipo_transaccion

    if segmento is not None:
        base_query += " AND segmento = :segmento"
        params["segmento"] = segmento

    if limit is not None:
        base_query += " LIMIT :limit"
        params["limit"] = limit

    query = text(base_query)

    start = perf_counter()
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params)
    elapsed = perf_counter() - start

    logger.info(
        "Loaded %d rows (tipo_transaccion=%s, segmento=%s) in %.3f s",
        len(df), tipo_transaccion, segmento, elapsed,
    )
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 # Radio de la Tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000 # distancia en metros

def fetch_comparable_listings(
    *,
    latitude: float,
    longitude: float,
    ciudad: str | None,
    pais: str | None,
    tipo_propiedad: str | None,
    segmento: str | None,
    tipo_transaccion: str | None,
    m2_construidos: float = 0,
    m2_terreno: float = 0,
    dormitorios: int = 0,
    banos: int = 0,
    precio_m2_referencia: float = 0.0,
    limit: int = 20,
) -> list[dict]:
    """
    Sistema inteligente de comparables basado en similitud multivariada.
    """
    # 1. Búsqueda con filtros flexibles usando radio geográfico aproximado (~5km max)
    query = text(
        """
        SELECT
            id_propiedad, tipo_propiedad, estado_propiedad, segmento, tipo_transaccion,
            ciudad, pais, latitude, longitude,
            m2_construidos, m2_terreno, dormitorios, banos,
            precio_publicacion, precio_venta, precio_alquiler_mes, precio_m2,
            tiempo_en_mercado, fecha_venta, numero_reducciones, cluster_zona
        FROM public.property_analytics
        WHERE (:ciudad IS NULL OR ciudad = :ciudad)
          AND (:pais IS NULL OR pais = :pais)
          AND (:segmento IS NULL OR segmento = :segmento)
          AND (:tipo_transaccion IS NULL OR tipo_transaccion = :tipo_transaccion)
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND ABS(latitude  - :lat0) <= 0.045
          AND ABS(longitude - :lon0) <= 0.045
          AND (precio_venta IS NOT NULL OR precio_publicacion IS NOT NULL OR precio_alquiler_mes IS NOT NULL)
        """
    )

    params = {
        "lat0": float(latitude),
        "lon0": float(longitude),
        "ciudad": ciudad,
        "pais": pais,
        "segmento": segmento,
        "tipo_transaccion": tipo_transaccion,
    }

    with get_connection() as conn:
        rows = conn.execute(query, params).mappings().all()

    candidates = [dict(r) for r in rows]
    if not candidates:
        return []

    area_ref = m2_construidos if m2_construidos > 0 else m2_terreno

    # Filtros adaptativos: si hay abundancia de comparables (>100), somos más estrictos con el área
    if len(candidates) > 100 and area_ref > 0:
        candidates = [
            c for c in candidates 
            if (float(c.get('m2_construidos') or 0) > 0 or float(c.get('m2_terreno') or 0) > 0) and 
               abs(float(c.get('m2_construidos') or c.get('m2_terreno') or 0) - area_ref) / area_ref <= 0.5
        ]

    # Eliminación de outliers de precio/m2 (Drop top 5% y bottom 5%)
    if len(candidates) >= 20:
        valid_prices = [float(c['precio_m2']) for c in candidates if c.get('precio_m2') and c['precio_m2'] > 0]
        if valid_prices:
            valid_prices.sort()
            lower_bound = valid_prices[max(0, int(len(valid_prices) * 0.05))]
            upper_bound = valid_prices[min(len(valid_prices) - 1, int(len(valid_prices) * 0.95))]
            candidates = [
                c for c in candidates 
                if c.get('precio_m2') and lower_bound <= float(c['precio_m2']) <= upper_bound
            ]

    # Constantes de los pesos
    W_DIST = 0.40
    W_AREA = 0.25
    W_PRICE = 0.25
    W_FEAT = 0.10

    results = []
    
    for c in candidates:
        comp_lat = float(c["latitude"])
        comp_lon = float(c["longitude"])
        dist_m = haversine_distance(latitude, longitude, comp_lat, comp_lon)
        
        # 1. Distancia Geográfica Real
        score_dist = max(0.0, 1.0 - (dist_m / 5000.0))
        
        # 2. Similitud de Área
        comp_area = float(c.get("m2_construidos") or 0) if (c.get("m2_construidos") or 0) > 0 else float(c.get("m2_terreno") or 0)
        if area_ref > 0 and comp_area > 0:
            diff_area_pct = abs(comp_area - area_ref) / area_ref
            score_area = max(0.0, 1.0 - diff_area_pct)
        else:
            score_area = 0.5
            
        # 3. Similitud de Precio
        comp_precio_m2 = float(c.get("precio_m2") or 0)
        if precio_m2_referencia > 0 and comp_precio_m2 > 0:
            diff_price_pct = abs(comp_precio_m2 - precio_m2_referencia) / precio_m2_referencia
            score_precio = max(0.0, 1.0 - diff_price_pct)
        else:
            score_precio = 0.5
            
        # 4. Similitud de Características
        comp_beds = int(c.get("dormitorios") or 0)
        comp_baths = int(c.get("banos") or 0)
        comp_tipo = c.get("tipo_propiedad")
        
        score_feats = 0.0
        max_feats = 1
        if tipo_propiedad and comp_tipo == tipo_propiedad:
            score_feats += 1
            
        if dormitorios > 0:
            max_feats += 1
            if comp_beds == dormitorios: score_feats += 1
            elif abs(comp_beds - dormitorios) == 1: score_feats += 0.5
            
        if banos > 0:
            max_feats += 1
            if comp_baths == banos: score_feats += 1
            elif abs(comp_baths - banos) == 1: score_feats += 0.5
            
        score_caract = score_feats / max_feats
        
        # RAW SCORE
        raw_score = (score_dist * W_DIST) + (score_area * W_AREA) + (score_precio * W_PRICE) + (score_caract * W_FEAT)
        
        # 5. Comportamiento del Mercado
        tiempo_mercado = int(c.get("tiempo_en_mercado") or 0)
        num_reds = int(c.get("numero_reducciones") or 0)
        
        bonus = 0.0
        if c.get("fecha_venta"):
            bonus += 0.10 # Valor de cierre (comparable real)
        elif tiempo_mercado < 30 and tiempo_mercado > 0:
            bonus += 0.05
        elif tiempo_mercado > 180:
            bonus -= 0.05
            
        if num_reds > 0:
            bonus -= (min(num_reds, 5) * 0.02)
            
        # Score_Total limitado a [0, 1]
        final_score = min(max(raw_score + bonus, 0.0), 1.0)
        
        # Explicabilidad para el Frontend
        diferencia_area_abs = abs(comp_area - area_ref) if area_ref > 0 and comp_area > 0 else 0
        msg_area = "Área casi igual" if diferencia_area_abs < (area_ref * 0.1) else f"Diferencia: ~{int(diferencia_area_abs)} m²"
        msg_dist = f"A {int(dist_m)} metros" if dist_m < 1000 else f"A {dist_m/1000:.1f} km"
        msg_sim = f"Muy similar ({int(final_score * 100)}%)" if final_score > 0.8 else f"Similitud: {int(final_score * 100)}%"
        
        c["similitud_score"] = float(final_score)
        c["similitud_explicacion"] = [msg_sim, msg_dist, msg_area]
        c["distancia_metros"] = float(dist_m)
        c["diferencia_area_m2"] = float(diferencia_area_abs)
        
        results.append(c)
        
    # 6. Ordenar por score de similitud
    results.sort(key=lambda x: x["similitud_score"], reverse=True)
    return results[:limit]


def fetch_nearest_zone_cluster(*, latitude: float, longitude: float) -> dict | None:
    """
    Busca el centroide más cercano en zona_clusters para inferir ciudad/pais
    cuando no vienen en el request.
    """
    query = text(
        """
        SELECT
            cluster_id,
            ciudad,
            pais,
            centroide_lat,
            centroide_lng
        FROM public.zona_clusters
        ORDER BY (
            (centroide_lat - :lat0) * (centroide_lat - :lat0)
            + (centroide_lng - :lon0) * (centroide_lng - :lon0)
        ) ASC
        LIMIT 1
        """
    )
    with get_connection() as conn:
        row = conn.execute(
            query, {"lat0": float(latitude), "lon0": float(longitude)}
        ).mappings().first()
    return dict(row) if row else None


def fetch_location_market_stats(
    *,
    latitude: float,
    longitude: float,
    tipo_transaccion: str,
    segmento: str,
    tipo_propiedad: str | None = None,
    ciudad: str | None = None,
    pais: str | None = None,
    min_cluster_samples: int = 8,
) -> dict | None:
    """
    Estima señales locales de mercado para requests sin precio_publicacion.

    Devuelve un precio_m2 de referencia dependiente de la ubicación para que
    el fallback sin precio reaccione mejor a cambios de coordenadas.
    """
    nearest = fetch_nearest_zone_cluster(latitude=latitude, longitude=longitude)

    resolved_city = ciudad or (str(nearest["ciudad"]) if nearest and nearest.get("ciudad") else None)
    resolved_country = pais or (str(nearest["pais"]) if nearest and nearest.get("pais") else None)
    cluster_id = int(nearest["cluster_id"]) if nearest and nearest.get("cluster_id") is not None else None

    if not resolved_city:
        return None

    target_col = "precio_alquiler_mes" if tipo_transaccion == "Alquiler" else "precio_venta"
    normalized_tipo = (tipo_propiedad or "").strip().lower()

    def _query_stats(*, use_cluster: bool, tipo_scope: str) -> dict | None:
        cluster_clause = "AND cluster_zona = :cluster_zona" if use_cluster else ""
        tipo_clause = ""
        if tipo_scope == "exact" and normalized_tipo:
            tipo_clause = "AND LOWER(TRIM(tipo_propiedad)) = :tipo_propiedad_norm"
        elif tipo_scope == "non_terrain":
            tipo_clause = "AND LOWER(TRIM(COALESCE(tipo_propiedad, ''))) <> 'terreno'"

        query = text(
            f"""
            SELECT
                COUNT(*) AS comparable_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY precio_m2) AS precio_m2_mediana,
                AVG(ratio_activas_vendidas_zona) AS ratio_activas_vendidas_zona_promedio
            FROM public.property_analytics
            WHERE tipo_transaccion = :tipo_transaccion
              AND segmento = :segmento
              AND ciudad = :ciudad
              AND (:pais IS NULL OR pais = :pais)
              {cluster_clause}
              {tipo_clause}
              AND precio_m2 IS NOT NULL
              AND precio_m2 > 0
              AND {target_col} IS NOT NULL
              AND {target_col} > 0
            """
        )
        params = {
            "tipo_transaccion": tipo_transaccion,
            "segmento": segmento,
            "ciudad": resolved_city,
            "pais": resolved_country,
        }
        if tipo_scope == "exact" and normalized_tipo:
            params["tipo_propiedad_norm"] = normalized_tipo
        if use_cluster:
            params["cluster_zona"] = cluster_id

        with get_connection() as conn:
            row = conn.execute(query, params).mappings().first()

        if not row:
            return None

        comparable_count = int(row["comparable_count"] or 0)
        precio_m2_mediana = row["precio_m2_mediana"]
        if comparable_count <= 0 or precio_m2_mediana is None:
            return None

        return {
            "scope": "cluster" if use_cluster else "city",
            "tipo_scope": tipo_scope,
            "tipo_propiedad": normalized_tipo or None,
            "ciudad": resolved_city,
            "pais": resolved_country,
            "cluster_zona": cluster_id if use_cluster else None,
            "comparable_count": comparable_count,
            "precio_m2_mediana": float(precio_m2_mediana),
            "ratio_activas_vendidas_zona": float(row["ratio_activas_vendidas_zona_promedio"] or 0.0),
        }

    search_scopes: list[str] = []
    if normalized_tipo:
        search_scopes.append("exact")
    if normalized_tipo != "terreno":
        search_scopes.append("non_terrain")
    search_scopes.append("all")

    seen_scopes: set[str] = set()
    for tipo_scope in search_scopes:
        if tipo_scope in seen_scopes:
            continue
        seen_scopes.add(tipo_scope)

        city_stats = _query_stats(use_cluster=False, tipo_scope=tipo_scope)

        if cluster_id is not None:
            cluster_stats = _query_stats(use_cluster=True, tipo_scope=tipo_scope)
            if cluster_stats and cluster_stats["comparable_count"] >= min_cluster_samples:
                if city_stats and city_stats.get("precio_m2_mediana"):
                    cluster_pm2 = float(cluster_stats["precio_m2_mediana"])
                    city_pm2 = float(city_stats["precio_m2_mediana"])
                    # Suavizamos extremos del cluster sin perder la señal local.
                    cluster_stats["precio_m2_mediana"] = math.sqrt(cluster_pm2 * city_pm2)
                return cluster_stats

        if city_stats:
            return city_stats

    return None

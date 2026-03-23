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
    Carga los datos desde Postgres, opcionalmente filtrados por
    tipo_transaccion y/o segmento para entrenar modelos especializados.
    """
    base_query = """
        SELECT
            id_propiedad,
            mlsid,
            tipo_propiedad,
            subtipo_original,
            categoria_propiedad,
            estado_propiedad,
            segmento,
            tipo_transaccion,
            latitude,
            longitude,
            cluster_zona,
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
            status,
            transaction_type,
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
        len(df),
        tipo_transaccion,
        segmento,
        elapsed,
    )

    return df


def fetch_comparable_listings(
    *,
    latitude: float,
    longitude: float,
    ciudad: str | None,
    pais: str | None,
    tipo_propiedad: str | None,
    segmento: str | None,
    tipo_transaccion: str | None,
    m2_construidos: float,
    m2_terreno: float,
    limit: int = 20,
) -> list[dict]:
    """
    Devuelve comparables desde la tabla, filtrados también por segmento
    y tipo_transaccion para mayor relevancia.
    """
    area = m2_construidos if m2_construidos and m2_construidos > 0 else m2_terreno
    min_area = area * 0.7 if area and area > 0 else 0
    max_area = area * 1.3 if area and area > 0 else 1e12

    query = text(
        """
        SELECT
            id_propiedad,
            mlsid,
            tipo_propiedad,
            segmento,
            tipo_transaccion,
            ciudad,
            pais,
            latitude,
            longitude,
            m2_construidos,
            m2_terreno,
            precio_publicacion,
            precio_venta,
            precio_alquiler_mes,
            precio_m2,
            fecha_venta,
            status,
            transaction_type
        FROM public.property_analytics
        WHERE
            (:ciudad IS NULL OR ciudad = :ciudad)
            AND (:pais IS NULL OR pais = :pais)
            AND (:tipo_propiedad IS NULL OR tipo_propiedad = :tipo_propiedad)
            AND (:segmento IS NULL OR segmento = :segmento)
            AND (:tipo_transaccion IS NULL OR tipo_transaccion = :tipo_transaccion)
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
            AND (
                (m2_construidos BETWEEN :min_area AND :max_area)
                OR (m2_terreno BETWEEN :min_area AND :max_area)
            )
            AND (precio_venta IS NOT NULL OR precio_publicacion IS NOT NULL OR precio_alquiler_mes IS NOT NULL)
            AND ABS(latitude - :lat0) <= 0.2
            AND ABS(longitude - :lon0) <= 0.2
        ORDER BY (ABS(latitude - :lat0) + ABS(longitude - :lon0)) ASC
        LIMIT :limit
        """
    )

    params = {
        "lat0": float(latitude),
        "lon0": float(longitude),
        "ciudad": ciudad,
        "pais": pais,
        "tipo_propiedad": tipo_propiedad,
        "segmento": segmento,
        "tipo_transaccion": tipo_transaccion,
        "min_area": float(min_area),
        "max_area": float(max_area),
        "limit": int(limit),
    }

    with get_connection() as conn:
        rows = conn.execute(query, params).mappings().all()
    return [dict(r) for r in rows]


def fetch_nearest_zone_cluster(*, latitude: float, longitude: float) -> dict | None:
    """
    Busca el centroide (cluster_id) más cercano a las coordenadas recibidas.
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
            query,
            {"lat0": float(latitude), "lon0": float(longitude)},
        ).mappings().first()

    return dict(row) if row else None
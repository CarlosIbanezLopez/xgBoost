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


def fetch_property_analytics(limit: int | None = None) -> pd.DataFrame:
    """
    Carga los datos crudos desde Postgres.

    Ajusta el nombre de la tabla / schema si es necesario.
    """
    base_query = """
        SELECT
            id_propiedad,
            mlsid,
            tipo_propiedad,
            subtipo_original,
            categoria_propiedad,
            estado_propiedad,
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
    """

    if limit is not None:
        base_query += " LIMIT :limit"

    query = text(base_query)

    start = perf_counter()
    with get_connection() as conn:
        params = {"limit": limit} if limit is not None else {}
        df = pd.read_sql(query, conn, params=params)
    elapsed = perf_counter() - start

    logger.info(
        "Loaded %d rows from property_analytics in %.3f s (limit=%s)",
        len(df),
        elapsed,
        str(limit),
    )

    return df


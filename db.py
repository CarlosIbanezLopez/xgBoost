from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text

from config import POSTGRES_DSN


engine = create_engine(POSTGRES_DSN)


@contextmanager
def get_connection():
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


def fetch_property_analytics() -> pd.DataFrame:
    """
    Carga los datos crudos desde Postgres.

    Ajusta el nombre de la tabla / schema si es necesario.
    """
    query = text(
        """
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
    )

    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    return df


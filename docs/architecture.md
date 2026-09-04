# Arquitectura

## Estilo

El servicio usa una arquitectura por capas dentro de un monolito modular. Esta
estructura mantiene un solo proceso desplegable, pero separa el transporte HTTP,
la orquestación, el entrenamiento de modelos y la infraestructura externa.

## Responsabilidades

| Capa | Ubicación | Responsabilidad |
| --- | --- | --- |
| Composición | `app/application.py` | Crear FastAPI y registrar routers |
| API | `app/api/` | Validar contratos, autenticación y exponer endpoints |
| Servicios | `app/services/` | Coordinar predicciones, entrenamiento y caché |
| ML | `app/ml/` | Preparar datos, entrenar y persistir artefactos |
| Infraestructura | `app/infrastructure/` | Conectar y consultar PostgreSQL |
| Core | `app/core/` | Leer configuración y resolver rutas de modelos |

La dirección principal de las llamadas es:

```text
HTTP -> API -> servicios -> ML / infraestructura -> PostgreSQL y models/
```

## Compatibilidad

Los módulos de la raíz son adaptadores deliberados:

- `main.py` conserva `uvicorn main:app` y los símbolos públicos anteriores.
- `config.py`, `db.py` y `ml_pipeline.py` redirigen a sus implementaciones bajo
  `app/` para mantener scripts e imports existentes.
- `.env` y `models/` siguen resolviéndose desde la raíz del repositorio.
- Los paths, métodos, esquemas y nombres de operación HTTP se mantienen.

## Límites operativos

- PostgreSQL solo se consulta al entrenar, enriquecer ubicaciones o buscar
  comparables; importar la aplicación no abre una conexión.
- Los modelos se cargan bajo demanda y quedan en caché en memoria.
- Un entrenamiento exitoso invalida la caché para que la próxima predicción
  cargue los artefactos nuevos.

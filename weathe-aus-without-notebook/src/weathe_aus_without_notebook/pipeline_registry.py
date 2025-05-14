"""
Registro de pipelines.
"""

from typing import Dict

from kedro.pipeline import Pipeline
# Corregir esta línea para usar el nombre real de tu paquete
from weathe_aus_without_notebook.pipelines import data_to_postgres, data_transform

def register_pipelines() -> Dict[str, Pipeline]:
    """
    Registra los pipelines del proyecto.
    
    Returns:
        Un diccionario con los pipelines.
    """
    data_to_postgres_pipeline = data_to_postgres.create_pipeline()
    data_transform_pipeline = data_transform.create_pipeline()
    
    return {
        "data_to_postgres": data_to_postgres_pipeline,
        "data_transform": data_transform_pipeline,
        "__default__": data_to_postgres_pipeline + data_transform_pipeline,
    }
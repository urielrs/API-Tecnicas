from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import os

# Obtener la ruta absoluta a la carpeta 'results'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'api', 'results')

class AnalysisResultsAPIView(APIView):
    """
    API para servir todos los resultados del análisis de datos
    """
    def get(self, request, format=None):
        data = {}
        # Lista de todos los resultados JSON que generaste
        files_to_read = [
            'df_head_10.json',
            'df_info.json',
            'calss_value_counts.json',
            'df_describe.json',
            'calss_corr_sorted.json',
            'X_corr_matrix.json',
            'feature_importances_top_20.json',
            'columns_top_10.json',
            'X_train_reduced_head_10.json'
        ]

        for filename in files_to_read:
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    # Usar el nombre del archivo (sin .json) como clave en la respuesta
                    key = filename.replace('.json', '')
                    data[key] = content
            except FileNotFoundError:
                return Response(
                    {"error": f"Archivo de resultado no encontrado: {filename}. Ejecuta 'python data_processor.py'"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except json.JSONDecodeError:
                 return Response(
                    {"error": f"Error al decodificar JSON en: {filename}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(data, status=status.HTTP_200_OK)
"""
GeoUtils - Geographic data utility module for handling KMZ, KML, and other geospatial formats.
"""

import zipfile
import tempfile
import os
import glob
import warnings
import re
import geopandas as gpd
import pandas as pd


class GeoUtils:
    """
    A utility class for handling geographic data operations, including:
    - Loading and processing KMZ files
    - Working with KML data
    - Geographic data transformations
    """

    @staticmethod
    def load_kmz(path, layer=None, verbose=False):
        """
        Extrai um arquivo .kmz (zip de .kml) e retorna um GeoDataFrame com os dados.
        Suporta múltiplos .kml dentro do .kmz (concatena-os).
        
        Args:
            path (str): Caminho para o arquivo .kmz
            layer (str, optional): Nome específico da camada/layer para carregar.
                                   Se None, carrega a camada padrão.
            verbose (bool): Se True, mostra avisos sobre múltiplas camadas.
                           Se False (padrão), suprime os avisos do pyogrio.
            
        Returns:
            GeoDataFrame: GeoDataFrame contendo os dados dos arquivos .kml
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            ValueError: Se não for um arquivo .kmz ou não contiver .kml
            
        Example:
            >>> gdf = GeoUtils.load_kmz("dados/meu_arquivo.kmz")
            >>> display(gdf.head())
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if not path.lower().endswith('.kmz'):
            raise ValueError("O arquivo deve ter extensão .kmz")
        
        gdfs = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(tmpdir)
            kml_files = glob.glob(os.path.join(tmpdir, '**', '*.kml'), recursive=True)
            if not kml_files:
                raise ValueError("Nenhum arquivo .kml encontrado dentro do .kmz")
            for kml in kml_files:
                try:
                    # Suprime avisos de pyogrio sobre múltiplas camadas se verbose=False
                    if not verbose:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, module='pyogrio')
                            if layer:
                                g = gpd.read_file(kml, driver='KML', layer=layer)
                            else:
                                g = gpd.read_file(kml, driver='KML')
                    else:
                        if layer:
                            g = gpd.read_file(kml, driver='KML', layer=layer)
                        else:
                            g = gpd.read_file(kml, driver='KML')
                except Exception:
                    # tentar sem especificar driver caso haja problemas
                    if not verbose:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, module='pyogrio')
                            g = gpd.read_file(kml)
                    else:
                        g = gpd.read_file(kml)
                gdfs.append(g)
        
        if not gdfs:
            return gpd.GeoDataFrame()
        if len(gdfs) == 1:
            return gdfs[0].reset_index(drop=True)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)).reset_index(drop=True)

    @staticmethod
    def load_kml(path):
        """
        Carrega um arquivo .kml e retorna um GeoDataFrame.
        
        Args:
            path (str): Caminho para o arquivo .kml
            
        Returns:
            GeoDataFrame: GeoDataFrame contendo os dados do arquivo .kml
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            ValueError: Se não for um arquivo .kml
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if not path.lower().endswith('.kml'):
            raise ValueError("O arquivo deve ter extensão .kml")
        
        try:
            gdf = gpd.read_file(path, driver='KML')
        except Exception:
            gdf = gpd.read_file(path)
        
        return gdf.reset_index(drop=True)

    @staticmethod
    def list_kmz_layers(path):
        """
        Lista todas as camadas (layers) disponíveis em um arquivo .kmz.
        
        Usa captura de avisos do pyogrio para extrair nomes de camadas.
        
        Args:
            path (str): Caminho para o arquivo .kmz
            
        Returns:
            dict: Dicionário com informações sobre os arquivos KML e suas camadas
            
        Example:
            >>> layers_info = GeoUtils.list_kmz_layers("dados/meu_arquivo.kmz")
            >>> for kml_file, layers in layers_info.items():
            ...     print(f"{kml_file}: {len(layers)} camadas")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if not path.lower().endswith('.kmz'):
            raise ValueError("O arquivo deve ter extensão .kmz")
        
        layers_info = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(tmpdir)
            kml_files = glob.glob(os.path.join(tmpdir, '**', '*.kml'), recursive=True)
            if not kml_files:
                raise ValueError("Nenhum arquivo .kml encontrado dentro do .kmz")
            
            for kml in kml_files:
                kml_name = os.path.basename(kml)
                layers = []
                try:
                    # Capture warning that contains layer names
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            # Try reading - this will trigger the warning
                            gdf = gpd.read_file(kml, driver='KML')
                            # If no warning, we got the default layer
                            layers.append('(default layer)')
                        except Exception:
                            pass
                        
                        # Extract layer names from warning message
                        for warning in w:
                            if 'More than one layer found' in str(warning.message):
                                warning_text = str(warning.message)
                                # Extract layer names using regex
                                # Pattern: 'Layer Name' (with quotes) or Layer Name: '...'
                                matches = re.findall(r"'([^']+)'", warning_text)
                                if matches:
                                    layers = matches
                                    break
                    
                    if not layers:
                        layers = ['(default layer)']
                        
                except Exception as e:
                    layers = [f'(error: {str(e)[:60]})']
                
                layers_info[kml_name] = layers
        
        return layers_info

    @staticmethod
    def get_column_info(gdf):
        """
        Retorna informações sobre as colunas de um GeoDataFrame.
        
        Args:
            gdf (GeoDataFrame): GeoDataFrame para análise
            
        Returns:
            dict: Dicionário com informações sobre cada coluna
        """
        return {
            'columns': list(gdf.columns),
            'non_null_counts': gdf.notna().sum().to_dict(),
            'dtypes': gdf.dtypes.to_dict(),
            'shape': gdf.shape
        }

    @staticmethod
    def filter_by_column(gdf, column, value=None):
        """
        Filtra um GeoDataFrame por coluna, removendo valores nulos ou filtrando por valor específico.
        
        Args:
            gdf (GeoDataFrame): GeoDataFrame para filtrar
            column (str): Nome da coluna
            value (optional): Valor específico para filtrar. Se None, remove apenas nulos.
            
        Returns:
            GeoDataFrame: GeoDataFrame filtrado
        """
        if value is None:
            return gdf[gdf[column].notna()].reset_index(drop=True)
        return gdf[gdf[column] == value].reset_index(drop=True)

import os
import mirdata
import numpy as np
import random
from typing import Dict, List, Tuple, Any

class DatasetManager:
    def __init__(self, data_home: str, datasets: List[str]):
        self.data_home = data_home
        self.datasets = datasets  # List of dataset names (strings)
        self.dataset_objs = {}    # Dictionary mapping name -> loaded mirdata dataset object
        self.tracks = {}          # Dictionary mapping name -> tracks dict
        self.valid_keys = {}      # Dictionary mapping name -> list of valid keys
        
    def initialize_datasets(self):
        """Initialize datasets"""
        dataset_configs = {
            'gtzan_genre': {
                'path': os.path.join(self.data_home, 'gtzan', 'gtzan_genre', 'audio', '22kmono'),
                'version': '1.0'
            },
            'beatles': {
                'path': os.path.join(self.data_home, 'beatles'),
                'version': None
            },
            'ballroom': {
                'path': os.path.join(self.data_home, 'ballroom'),
                'version': None
            },
            'rwc_popular': {
                'path': os.path.join(self.data_home, 'RWC-Popular'),
                'version': None
            },
            'rwc_jazz': {
                'path': os.path.join(self.data_home, 'RWC-Jazz'),
                'version': None
            },
            'rwc_classical': {
                'path': os.path.join(self.data_home, 'RWC-Classical'),
                'version': None
            }
        }
        
        # Load each dataset based on the provided configuration
        for dataset_name in self.datasets:
            if dataset_name not in dataset_configs:
                print(f"⚠ Unknown dataset: {dataset_name}")
                continue
                
            config = dataset_configs[dataset_name]
            try:
                print(f"Loading {dataset_name} dataset...")
                if config['version']:
                    dataset = mirdata.initialize(dataset_name, 
                                               version=config['version'], 
                                               data_home=config['path'])
                else:
                    dataset = mirdata.initialize(dataset_name, 
                                               data_home=config['path'])
                
                self.dataset_objs[dataset_name] = dataset
                self.tracks[dataset_name] = dataset.load_tracks()
                self.valid_keys[dataset_name] = self._get_valid_keys(dataset_name)
                print(f"✓ {dataset_name}: {len(self.valid_keys[dataset_name])} valid tracks")
                
            except Exception as e:
                print(f"✗ Failed to load {dataset_name}: {e}")
                continue
    
    def _get_valid_keys(self, dataset_name: str) -> List[str]:
        """Extract valid keys for a dataset"""
        """The datasets are manually checked for missing or invalid tracks.
        The valid keys are those without missing or invalid annotations"""

        tracks = self.tracks[dataset_name]
        
        # Identify keys with no or empty beat annotations
        none_keys = set()

        for k in tracks:
            ct = tracks[k]
            
            if ct.beats is None:
                none_keys.add(k)
                continue

            beat_times = ct.beats.times
            beat_pos = ct.beats.positions

            if (beat_times is None or (isinstance(beat_times, np.ndarray) and beat_times.size == 0) or
                beat_pos is None or (isinstance(beat_pos, np.ndarray) and beat_pos.size == 0)):
                none_keys.add(k)

        # Valid keys are all others
        valid_keys = [k for k in tracks if k not in none_keys]
        return valid_keys

    def get_all_valid_keys(self) -> List[Tuple[str, str]]:
        """Get all valid keys as (dataset_name, key) tuples"""
        all_valid_keys = []

        for dataset_name, keys in self.valid_keys.items():
            for key in keys:
                all_valid_keys.append((dataset_name, key))

        return all_valid_keys
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded datasets"""
        summary = {}
        total_valid = 0
        
        for dataset_name in self.dataset_objs.keys():
            valid_count = len(self.valid_keys[dataset_name])
            total_count = len(self.tracks[dataset_name])
            summary[dataset_name] = {
                'total_tracks': total_count,
                'valid_tracks': valid_count,
                'valid_ratio': valid_count / total_count if total_count > 0 else 0
            }
            total_valid += valid_count
        
        summary['overall'] = {
            'total_valid_tracks': total_valid,
            'datasets_loaded': len(self.dataset_objs)
        }
        
        return summary
    
    def print_split_distribution(self, keys: List[Tuple[str, str]], split_name: str):
        """Print the distribution of datasets in a split
        
        Args:
            keys: List of (dataset_name, key) tuples
            split_name: Name of the split (e.g., "Training", "Validation", "Test")
        """
        dataset_counts = {}
        for dataset_name, _ in keys:
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        print(f"{split_name} distribution:")
        for dataset, count in sorted(dataset_counts.items()):
            percentage = (count / len(keys)) * 100
            print(f"  {dataset}: {count} tracks ({percentage:.1f}%)")
        
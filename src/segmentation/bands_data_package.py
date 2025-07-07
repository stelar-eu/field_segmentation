from typing import List, Tuple, Optional, Dict
import glob
import os
from stelar_spatiotemporal.lib import get_filesystem

class BandDataPackage:
    BAND_NAME: str = None
    BAND_DIR: str = None
    path_pairs: List[Tuple[str, str]] = None
    file_extension:str = None

    def __init__(self, band_name, band_dir=None, file_extension:str = "RAS", path_list: Optional[List[Tuple[str, str]]] = None):
        self.BAND_NAME = band_name
        self.file_extension = file_extension

        if path_list is not None:
            # Initialize from provided path pairs
            self.path_pairs = path_list
            # Set BAND_DIR to the directory of the first file if provided
            if len(path_list) > 0:
                self.BAND_DIR = os.path.dirname(path_list[0][0])
            else:
                self.BAND_DIR = None
        elif band_dir is not None:
            # Initialize using directory
            self.BAND_DIR = band_dir
            self.validate_file_extension()

            if file_extension == "RAS":
                self.discover_ras_rhd_pairs()
        else:
            raise ValueError("Either band_dir or path_list must be provided")

    """
    Check if the band directory contains at least one file with the given file extension
    """
    def validate_file_extension(self):
        filesystem = get_filesystem(self.BAND_DIR)
        if len(filesystem.glob(os.path.join(self.BAND_DIR, "**", f'*.{self.file_extension}'))) == 0:
            raise ValueError(f"No files with extension .{self.file_extension} found in {self.BAND_DIR}")

    """
    Discover and pair RAS and RHD files in the band directory
    """
    def discover_ras_rhd_pairs(self):
        filesystem = get_filesystem(self.BAND_DIR)

        self.path_pairs = []
        ras_paths = filesystem.glob(os.path.join(self.BAND_DIR, "**", '*.RAS'))
        ras_bases = [os.path.basename(path).replace('.RAS', '') for path in ras_paths]

        rhd_paths = filesystem.glob(os.path.join(self.BAND_DIR, "**", '*.RHD'))
        rhd_bases = [os.path.basename(path).replace('.RHD', '') for path in rhd_paths]
        
        for ras_id, ras_base in enumerate(ras_bases):
            rhd_base = ras_base.replace('.RAS', '.RHD')
            try:
                rhd_id = rhd_bases.index(rhd_base)
                self.path_pairs.append((ras_paths[ras_id], rhd_paths[rhd_id]))
            except ValueError:
                raise ValueError(f"RHD file for {ras_base} not found")

class BandsDataPackage:
    B2_package: BandDataPackage = None
    B3_package: BandDataPackage = None
    B4_package: BandDataPackage = None
    B8_package: BandDataPackage = None

    def __init__(self, b2_dir=None, b3_dir=None, b4_dir=None, b8_dir=None, file_extension:str = "RAS", 
                 b2_paths=None, b3_paths=None, b4_paths=None, b8_paths=None):
        if b2_paths is not None or b3_paths is not None or b4_paths is not None or b8_paths is not None:
            # Initialize from provided paths
            self.B2_package = BandDataPackage("B2", path_list=b2_paths, file_extension=file_extension)
            self.B3_package = BandDataPackage("B3", path_list=b3_paths, file_extension=file_extension)
            self.B4_package = BandDataPackage("B4", path_list=b4_paths, file_extension=file_extension)
            self.B8_package = BandDataPackage("B8A", path_list=b8_paths, file_extension=file_extension)
        elif b2_dir is not None and b3_dir is not None and b4_dir is not None and b8_dir is not None:
            # Initialize from directories
            self.B2_package = BandDataPackage("B2", b2_dir, file_extension)
            self.B3_package = BandDataPackage("B3", b3_dir, file_extension)
            self.B4_package = BandDataPackage("B4", b4_dir, file_extension)
            self.B8_package = BandDataPackage("B8A", b8_dir, file_extension)
        else:
            raise ValueError("Either directory paths or file paths must be provided for all bands")

    @classmethod
    def from_file_list(cls, input_files: List[str], file_extension: str = "RAS") -> 'BandsDataPackage':
        """
        Create a BandsDataPackage from a list of files by grouping them by band and pairing RAS/RHD files.
        
        Parameters:
        -----------
        input_files : List[str]
            List of file paths (both RAS and RHD files)
        file_extension : str
            File extension to use (default: "RAS")
            
        Returns:
        --------
        BandsDataPackage
            A BandsDataPackage object with all the band packages initialized
        """
        # Group the files by band
        band_files = {
            "B2": [path for path in input_files if "B2" in os.path.basename(path)],
            "B3": [path for path in input_files if "B3" in os.path.basename(path)],
            "B4": [path for path in input_files if "B4" in os.path.basename(path)],
            "B8": [path for path in input_files if "BA8" in path or "B8" in os.path.basename(path)],
        }
        
        # Create path pairs for each band
        path_pairs = {}
        
        for band, files in band_files.items():
            path_pairs[band] = cls._create_path_pairs(files, file_extension)
            
        # Check if we found files for all bands
        if not all(path_pairs.values()):
            raise ValueError(f"Could not find {file_extension}/RHD pairs for all required bands (B2, B3, B4, B8/BA8)")
        
        # Create BandsDataPackage from the path pairs
        return cls(
            file_extension=file_extension,
            b2_paths=path_pairs["B2"],
            b3_paths=path_pairs["B3"],
            b4_paths=path_pairs["B4"],
            b8_paths=path_pairs["B8"]
        )
    
    @staticmethod
    def _create_path_pairs(file_list: List[str], file_extension: str) -> List[Tuple[str, str]]:
        """
        Create pairs of (RAS, RHD) or other file extension pairs from a list of files.
        
        Parameters:
        -----------
        file_list : List[str]
            List of file paths
        file_extension : str
            The main file extension (e.g., "RAS")
            
        Returns:
        --------
        List[Tuple[str, str]]
            List of tuples, each containing a pair of file paths (primary_file, header_file)
        """
        path_pairs = []
        
        primary_files = [path for path in file_list if path.upper().endswith(f".{file_extension.upper()}")]
        header_files = [path for path in file_list if path.upper().endswith(".RHD")]
        
        # Match primary and header files by base name
        for primary_path in primary_files:
            primary_base = os.path.basename(primary_path).upper().replace(f'.{file_extension.upper()}', '')
            matching_header = next(
                (hdr for hdr in header_files 
                 if os.path.basename(hdr).upper().replace('.RHD', '') == primary_base), 
                None
            )
            
            if matching_header:
                path_pairs.append((primary_path, matching_header))
                
        return path_pairs
    
    def tolist(self):
        return [self.B2_package, self.B3_package, self.B4_package, self.B8_package]
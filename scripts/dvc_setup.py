"""
DVC Setup and Management Script
Handles Data Version Control initialization and operations
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVCManager:
    """Class for managing DVC operations"""
    
    def __init__(self, project_root: str):
        """
        Initialize DVCManager
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.dvc_dir = self.project_root / ".dvc"
        
    def is_dvc_initialized(self) -> bool:
        """
        Check if DVC is initialized in the project
        
        Returns:
            True if DVC is initialized
        """
        return self.dvc_dir.exists()
        
    def initialize_dvc(self) -> bool:
        """
        Initialize DVC in the project
        
        Returns:
            True if successful
        """
        try:
            if self.is_dvc_initialized():
                logger.info("DVC already initialized")
                return True
                
            logger.info("Initializing DVC...")
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("DVC initialized successfully")
                return True
            else:
                logger.error(f"DVC initialization failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing DVC: {e}")
            return False
            
    def add_local_remote(self, remote_path: str, remote_name: str = "localstorage") -> bool:
        """
        Add a local remote storage for DVC
        
        Args:
            remote_path: Path to local remote storage
            remote_name: Name for the remote
            
        Returns:
            True if successful
        """
        try:
            # Create remote directory if it doesn't exist
            os.makedirs(remote_path, exist_ok=True)
            
            logger.info(f"Adding local remote: {remote_name} at {remote_path}")
            result = subprocess.run(
                ["dvc", "remote", "add", "-d", remote_name, remote_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Local remote added successfully")
                return True
            else:
                # Remote might already exist, try to modify
                logger.info("Remote exists, updating...")
                result = subprocess.run(
                    ["dvc", "remote", "modify", remote_name, "url", remote_path],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info("Local remote updated successfully")
                    return True
                else:
                    logger.error(f"Failed to add/update remote: {result.stderr}")
                    return False
                
        except Exception as e:
            logger.error(f"Error adding local remote: {e}")
            return False
            
    def add_data_to_dvc(self, data_path: str) -> bool:
        """
        Add data file/directory to DVC tracking
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Adding {data_path} to DVC...")
            result = subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"{data_path} added to DVC successfully")
                return True
            else:
                logger.error(f"Failed to add to DVC: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding to DVC: {e}")
            return False
            
    def push_to_remote(self) -> bool:
        """
        Push DVC tracked files to remote storage
        
        Returns:
            True if successful
        """
        try:
            logger.info("Pushing data to remote...")
            result = subprocess.run(
                ["dvc", "push"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Data pushed to remote successfully")
                return True
            else:
                logger.error(f"Failed to push to remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing to remote: {e}")
            return False
            
    def pull_from_remote(self) -> bool:
        """
        Pull DVC tracked files from remote storage
        
        Returns:
            True if successful
        """
        try:
            logger.info("Pulling data from remote...")
            result = subprocess.run(
                ["dvc", "pull"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Data pulled from remote successfully")
                return True
            else:
                logger.error(f"Failed to pull from remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling from remote: {e}")
            return False
            
    def get_dvc_status(self) -> Optional[str]:
        """
        Get DVC status
        
        Returns:
            Status string or None if error
        """
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Failed to get DVC status: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting DVC status: {e}")
            return None
            
    def create_data_version(self, version_name: str) -> bool:
        """
        Create a new data version with git tag
        
        Args:
            version_name: Name for the version (e.g., 'v1.0', 'baseline')
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating data version: {version_name}")
            result = subprocess.run(
                ["git", "tag", "-a", version_name, "-m", f"Data version {version_name}"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Version {version_name} created successfully")
                return True
            else:
                logger.error(f"Failed to create version: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            return False


def setup_dvc_pipeline(project_root: str, data_file: str, 
                       local_storage: str = "/tmp/dvc-storage") -> bool:
    """
    Complete DVC setup pipeline
    
    Args:
        project_root: Root directory of the project
        data_file: Path to data file to track
        local_storage: Path for local remote storage
        
    Returns:
        True if successful
    """
    manager = DVCManager(project_root)
    
    # Step 1: Initialize DVC
    if not manager.initialize_dvc():
        return False
        
    # Step 2: Add local remote
    if not manager.add_local_remote(local_storage):
        return False
        
    # Step 3: Add data to DVC
    if os.path.exists(data_file):
        if not manager.add_data_to_dvc(data_file):
            return False
    else:
        logger.warning(f"Data file not found: {data_file}")
        
    # Step 4: Push to remote
    if not manager.push_to_remote():
        return False
        
    logger.info("DVC setup completed successfully!")
    return True


if __name__ == "__main__":
    # Example usage
    # setup_dvc_pipeline(
    #     project_root=".",
    #     data_file="data/insurance_data.txt",
    #     local_storage="/path/to/storage"
    # )
    pass

import os
import subprocess
import time
import json
import requests
import signal
import atexit
import logging
from typing import Optional, List, Dict, Any, Union

class OllamaManager:
    """
    A class to manage Ollama models and server operations.
    Handles downloading models and running the Ollama server in the background.
    """
    
    def __init__(self, 
                 server_url: str = "http://localhost:11434",
                 ollama_path: str = "ollama",
                 log_level: int = logging.INFO):
        """
        Initialize the OllamaManager.
        
        Args:
            server_url: URL where the Ollama server will be accessible
            ollama_path: Path to the ollama executable
            log_level: Logging level (default: INFO)
        """
        self.server_url = server_url
        self.ollama_path = ollama_path
        self.server_process = None
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("OllamaManager")
        
        # Register cleanup function
        atexit.register(self.stop_server)
    
    def start_server(self, wait_for_ready: bool = True, timeout: int = 30) -> None:
        """
        Start the Ollama server in the background.
        
        Args:
            wait_for_ready: Whether to wait until the server is responding
            timeout: Maximum time to wait for server to become ready (in seconds)
        """
        if self.is_server_running():
            self.logger.info("Ollama server is already running")
            return
        
        try:
            self.logger.info("Starting Ollama server...")
            # Use DETACHED_PROCESS on Windows to prevent console window
            if os.name == 'nt':
                self.server_process = subprocess.Popen(
                    [self.ollama_path, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.DETACHED_PROCESS
                )
            else:
                self.server_process = subprocess.Popen(
                    [self.ollama_path, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            self.logger.info(f"Ollama server started with PID {self.server_process.pid}")
            
            if wait_for_ready:
                self._wait_for_server(timeout)
        
        except Exception as e:
            self.logger.error(f"Failed to start Ollama server: {str(e)}")
            raise
    
    def _wait_for_server(self, timeout: int = 30) -> None:
        """
        Wait for the server to become responsive.
        
        Args:
            timeout: Maximum time to wait (in seconds)
        
        Raises:
            TimeoutError: If the server does not respond within the timeout period
        """
        self.logger.info(f"Waiting for Ollama server to become ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(f"{self.server_url}/api/tags")
                if response.status_code == 200:
                    self.logger.info("Ollama server is ready")
                    return
            except requests.RequestException:
                pass
            
            time.sleep(1)
        
        self.stop_server()
        raise TimeoutError("Ollama server failed to start within the specified timeout")
    
    def stop_server(self) -> None:
        """
        Stop the Ollama server if it's running.
        """
        if self.server_process is not None:
            self.logger.info(f"Stopping Ollama server (PID {self.server_process.pid})...")
            
            try:
                if os.name == 'nt':
                    # On Windows use taskkill to ensure child processes are terminated
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.server_process.pid)])
                else:
                    # On Unix, send SIGTERM
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Server didn't terminate gracefully, forcing...")
                if os.name != 'nt':  # Already forced on Windows
                    self.server_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping server: {str(e)}")
            
            self.server_process = None
            self.logger.info("Ollama server stopped")
    
    def is_server_running(self) -> bool:
        """
        Check if the Ollama server is running and responsive.
        
        Returns:
            True if the server is running and responding, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def download_model(self, model_name: str, show_progress: bool = True) -> bool:
        """
        Download a model from Ollama.
        
        Args:
            model_name: Name of the model to download
            show_progress: Whether to show download progress
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Downloading model: {model_name}")
        
        # Make sure server is running
        server_was_started = False
        if not self.is_server_running():
            self.start_server()
            server_was_started = True
        
        try:
            if show_progress:
                # Use subprocess with real-time output for better progress visibility
                process = subprocess.Popen(
                    [self.ollama_path, "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        self.logger.info(line.strip())
                
                process.wait()
                success = process.returncode == 0
            else:
                # Silent mode
                result = subprocess.run(
                    [self.ollama_path, "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                success = result.returncode == 0
            
            if success:
                self.logger.info(f"Successfully downloaded model: {model_name}")
            else:
                self.logger.error(f"Failed to download model: {model_name}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")
            return False
        finally:
            # Stop the server if we started it
            if server_was_started:
                self.stop_server()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        server_was_started = False
        if not self.is_server_running():
            self.start_server()
            server_was_started = True
        
        try:
            response = requests.get(f"{self.server_url}/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
            else:
                self.logger.error(f"Failed to list models: {response.status_code} {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
        finally:
            if server_was_started:
                self.stop_server()
    
    def generate(self, 
                 model: str, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 stream: bool = False,
                 options: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a response from a model.
        
        Args:
            model: Name of the model to use
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            options: Additional options to pass to the model
            
        Returns:
            Response from the model (either a single dict or list of response chunks)
        """
        server_was_started = False
        if not self.is_server_running():
            self.start_server()
            server_was_started = True
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if options:
                payload.update(options)
            
            endpoint = f"{self.server_url}/api/generate"
            
            if stream:
                responses = []
                with requests.post(endpoint, json=payload, stream=True) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            responses.append(chunk)
                            if chunk.get('done', False):
                                break
                return responses
            else:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
        finally:
            if server_was_started:
                self.stop_server()

import os
import subprocess
import time
import json
import requests
import signal
import atexit
from typing import Optional, List, Dict, Any, Union

class OllamaManager:
    def __init__(self, server_url: str = "http://localhost:11434",ollama_path: str = "ollama"):
        self.server_url = server_url
        self.ollama_path = ollama_path
        self.server_process = None
        atexit.register(self.stop_server)
    
    def start_server(self, wait_for_ready: bool = True, timeout: int = 30) -> None:
        if self.is_server_running():
            print("Ollama server is already running", flush=True)
            return
        try:
            print("Starting Ollama server...", flush=True)
            
            self.server_process = subprocess.Popen(
                [self.ollama_path, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Ollama server started with PID {self.server_process.pid}", flush=True)
            if wait_for_ready:
                self._wait_for_server(timeout)
        
        except Exception as e:
            print(f"Failed to start Ollama server: {str(e)}", flush=True)
            raise
    
    def _wait_for_server(self, timeout: int = 30) -> None:
        print(f"Waiting for Ollama server to become ready (timeout: {timeout}s)...", flush=True)
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(f"{self.server_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    print("Ollama server is ready", flush=True)
                    return
            except requests.RequestException:
                pass
            
            time.sleep(1)
        
        print("Timeout waiting for server to start, attempting to stop", flush=True)
        self.stop_server()
        raise TimeoutError("Ollama server failed to start within the specified timeout")
    
    def stop_server(self) -> None:
        if self.server_process is not None:
            print(f"Stopping Ollama server (PID {self.server_process.pid})...", flush=True)
            try:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Server didn't terminate gracefully, forcing...", flush=True)
                    self.server_process.kill()
                    try:
                        self.server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("Server still didn't terminate after kill, using SIGKILL...", flush=True)
                        os.kill(self.server_process.pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error stopping server: {str(e)}", flush=True)
            
            try:
                if self.server_process.poll() is None:
                    print("Warning: Server process still appears to be running after stop attempt", flush=True)
            except:
                pass
            
            self.server_process = None
            
            time.sleep(1)
            if self.is_server_running():
                print("Warning: Server still responding after stop attempt", flush=True)
            else:
                print("Ollama server stopped successfully", flush=True)
    
    def is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
            
    def download_model(self, model_name: str, show_progress: bool = True) -> bool:
        print(f"Downloading model: {model_name}", flush=True)
        server_was_started = False
        if not self.is_server_running():
            print("Starting server for model download", flush=True)
            self.start_server()
            server_was_started = True
        
        try:
            print(f"Running ollama pull {model_name}", flush=True)
            result = subprocess.run(
                [self.ollama_path, "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False,
                timeout=600
            )
            success = result.returncode == 0
            if success:
                print(f"Successfully downloaded model: {model_name}", flush=True)
            else:
                print(f"Failed to download model: {model_name} (return code: {result.returncode})", flush=True)
            
            models = self.list_models()
            print(models,flush=True)
            model_exists = any(model.get("name") == model_name for model in models)
            print(f"Model verification: {model_name} exists: {model_exists}", flush=True)
            
            return model_exists
        
        except subprocess.TimeoutExpired:
            print(f"Timeout while downloading model {model_name}. This may indicate network issues or an extremely large model.", flush=True)
            return False
        except Exception as e:
            print(f"Error downloading model: {str(e)}", flush=True)
            import traceback
            print(f"Stack trace: {traceback.format_exc()}", flush=True)
            return False
            
    def list_models(self) -> List[Dict[str, Any]]:
        print("Listing available models...", flush=True)
        server_was_started = False
        if not self.is_server_running():
            print("Starting server to list models", flush=True)
            self.start_server()
            server_was_started = True
        
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"Found {len(models)} models", flush=True)
                return models
            else:
                print(f"Failed to list models: {response.status_code} {response.text}", flush=True)
                return []
        except Exception as e:
            print(f"Error listing models: {str(e)}", flush=True)
            return []
    

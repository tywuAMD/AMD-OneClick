"""
Configuration settings for AMD OneClick Notebook Manager
"""
import os
from typing import Optional


def _parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # K8s Configuration
    K8S_NAMESPACE: str = os.getenv("K8S_NAMESPACE", "default")
    
    # Default Notebook Image
    DEFAULT_IMAGE: str = os.getenv(
        "DEFAULT_IMAGE", 
        "docker.io/rocm/vllm:rocm7.12.0_gfx110X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"
    )
    
    # Available Images (can be extended)
    AVAILABLE_IMAGES: list = [
        "docker.io/rocm/vllm:rocm7.12.0_gfx110X-all_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0",
    ]
    
    # Notebook Configuration
    NOTEBOOK_TOKEN: str = os.getenv("NOTEBOOK_TOKEN", "amd-oneclick")
    NOTEBOOK_PORT: int = 8888
    NOTEBOOK_LABEL_PREFIX: str = "amd-oneclick"
    NOTEBOOK_NODE_HOSTNAME: Optional[str] = os.getenv("NOTEBOOK_NODE_HOSTNAME")
    PUBLIC_MODELS_HOST_PATH: str = os.getenv("PUBLIC_MODELS_HOST_PATH", "/models")
    PUBLIC_MODELS_MOUNT_PATH: str = os.getenv("PUBLIC_MODELS_MOUNT_PATH", "/models")
    
    # Resource Limits
    CPU_LIMIT: str = os.getenv("CPU_LIMIT", "128")
    MEMORY_LIMIT: str = os.getenv("MEMORY_LIMIT", "128Gi")
    GPU_LIMIT: str = os.getenv("GPU_LIMIT", "1")
    CPU_REQUEST: str = os.getenv("CPU_REQUEST", "40")
    MEMORY_REQUEST: str = os.getenv("MEMORY_REQUEST", "48Gi")
    
    # Cleanup Configuration
    IDLE_TIMEOUT_MINUTES: int = int(os.getenv("IDLE_TIMEOUT_MINUTES", "10"))
    MAX_LIFETIME_HOURS: int = int(os.getenv("MAX_LIFETIME_HOURS", "6"))
    
    # Email Configuration (optional)
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "noreply@amd-oneclick.local")
    
    # Service Configuration
    SERVICE_HOST: str = os.getenv("SERVICE_HOST", "localhost")
    NODE_PORT_BASE: int = int(os.getenv("NODE_PORT_BASE", "30000"))

    # Reservation integration
    RESERVATION_API_BASE_URL: str = os.getenv("RESERVATION_API_BASE_URL", "http://localhost:4100/api")
    NOTEBOOK_BRIDGE_SHARED_TOKEN: Optional[str] = os.getenv("NOTEBOOK_BRIDGE_SHARED_TOKEN")
    ALLOW_PUBLIC_NOTEBOOK_API: bool = _parse_bool("ALLOW_PUBLIC_NOTEBOOK_API", True)
    
    # PyPI Mirror for China
    PYPI_MIRROR: str = "https://pypi.tuna.tsinghua.edu.cn/simple"
    PYPI_HOST: str = "pypi.tuna.tsinghua.edu.cn"
    PYPI_HOST_IP: str = "101.6.15.130"
    
    # Admin Configuration
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")


settings = Settings()

"""
Kubernetes client for managing notebook instances
"""
import hashlib
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .config import settings

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SERVICE_ACCOUNT_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _normalize_platform_key(value: Optional[str]) -> str:
    return "".join(char for char in (value or "").strip().lower() if char.isalnum())


class K8sClient:
    """Kubernetes client for notebook management"""
    
    def __init__(self):
        """Initialize K8s client"""
        self._service_account_token: Optional[str] = None
        has_service_account_token = os.path.exists(SERVICE_ACCOUNT_TOKEN_PATH)
        has_kubernetes_host = bool(os.getenv("KUBERNETES_SERVICE_HOST"))

        if has_service_account_token:
            # Always use manual in-cluster config to avoid token formatting ambiguity
            # from library defaults (for example newline/prefix handling).
            self._load_manual_incluster_config()
            logger.info("Loaded in-cluster K8s config (manual token mode)")
        elif has_kubernetes_host:
            # We appear to be in-cluster but token is missing: fail fast to avoid anonymous access.
            raise RuntimeError(
                "KUBERNETES_SERVICE_HOST is set but service-account token is missing. "
                "Ensure automountServiceAccountToken is enabled."
            )
        else:
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig file")
            except config.ConfigException as error:
                raise RuntimeError(
                    "Kubernetes config not found for local development. "
                    "Set up kubeconfig or run inside a cluster."
                ) from error
        
        api_client = client.ApiClient()
        if self._service_account_token:
            # Force auth header at client level to avoid generator auth_settings mismatches.
            api_client.default_headers["authorization"] = f"Bearer {self._service_account_token}"
            logger.info("Configured explicit bearer token on Kubernetes ApiClient headers")

        self.core_v1 = client.CoreV1Api(api_client)
        self.apps_v1 = client.AppsV1Api(api_client)
        self.namespace = settings.K8S_NAMESPACE

    def _load_manual_incluster_config(self):
        """Build Kubernetes client config directly from mounted service-account token."""
        if not os.path.exists(SERVICE_ACCOUNT_TOKEN_PATH):
            raise RuntimeError("Service-account token file is missing.")

        with open(SERVICE_ACCOUNT_TOKEN_PATH, "r", encoding="utf-8") as token_file:
            token = token_file.read().strip()
        if not token:
            raise RuntimeError("Service-account token file is empty.")
        self._service_account_token = token

        host = os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        host = host.replace("https://", "").replace("http://", "")
        port = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS") or os.getenv("KUBERNETES_SERVICE_PORT", "443")

        configuration = client.Configuration.get_default_copy()
        configuration.host = f"https://{host}:{port}"
        configuration.verify_ssl = True
        if os.path.exists(SERVICE_ACCOUNT_CA_PATH):
            configuration.ssl_ca_cert = SERVICE_ACCOUNT_CA_PATH
        # Write a complete bearer value directly so request auth cannot depend
        # on api_key_prefix behavior across kubernetes-client versions.
        configuration.api_key = {"authorization": f"Bearer {token}"}
        configuration.api_key_prefix = {}
        client.Configuration.set_default(configuration)
    
    def _generate_instance_id(self, email: str) -> str:
        """Generate a unique instance ID from email"""
        hash_str = hashlib.md5(email.lower().encode()).hexdigest()[:8]
        return f"nb-{hash_str}"
    
    def _get_labels(self, email: str, instance_id: str) -> dict:
        """Generate labels for K8s resources"""
        return {
            "app": settings.NOTEBOOK_LABEL_PREFIX,
            "instance-id": instance_id,
            "email-hash": hashlib.md5(email.lower().encode()).hexdigest()[:16],
        }

    def _parse_iso_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp into a timezone-aware datetime."""
        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            logger.warning("Invalid ISO timestamp: %s", value)
            return None

    def _resolve_notebook_node_hostname(self, platform: Optional[str] = None) -> Optional[str]:
        """Resolve a reservation platform name to the Kubernetes node hostname."""
        normalized_platform = (platform or "").strip()
        if normalized_platform:
            platform_key = _normalize_platform_key(normalized_platform)
            mapped_hostname = settings.NOTEBOOK_PLATFORM_NODE_MAP.get(platform_key)
            if mapped_hostname:
                return mapped_hostname

            if settings.NOTEBOOK_PLATFORM_NODE_MAP:
                raise ValueError(f'No notebook node mapping configured for platform "{normalized_platform}".')

        return settings.NOTEBOOK_NODE_HOSTNAME
    
    def _get_pod_manifest(self, email: str, instance_id: str, image: str, 
                          github_info: Optional[dict] = None,
                          reservation_end_at: Optional[str] = None,
                          owner_username: Optional[str] = None,
                          platform: Optional[str] = None) -> dict:
        """Generate Pod manifest"""
        labels = self._get_labels(email, instance_id)
        target_node_hostname = self._resolve_notebook_node_hostname(platform)
        
        annotations = {
            "amd-oneclick/email": email,
            "amd-oneclick/created-at": datetime.now(timezone.utc).isoformat(),
        }

        normalized_owner_username = (owner_username or "").strip()
        if normalized_owner_username:
            annotations["amd-oneclick/owner-username"] = normalized_owner_username

        if reservation_end_at:
            annotations["amd-oneclick/reservation-end-at"] = reservation_end_at

        normalized_platform = (platform or "").strip()
        if normalized_platform:
            annotations["amd-oneclick/platform"] = normalized_platform

        if target_node_hostname:
            annotations["amd-oneclick/target-node-hostname"] = target_node_hostname
        
        # Add GitHub info to annotations if provided
        if github_info:
            annotations["amd-oneclick/github-org"] = github_info.get("org", "")
            annotations["amd-oneclick/github-repo"] = github_info.get("repo", "")
            annotations["amd-oneclick/github-branch"] = github_info.get("branch", "")
            annotations["amd-oneclick/github-path"] = github_info.get("path", "")
            annotations["amd-oneclick/github-raw-url"] = github_info.get("raw_url", "")

        tolerations = [
            {
                "key": "amd.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule"
            }
        ]
        if settings.NOTEBOOK_TOLERATE_UNSCHEDULABLE:
            tolerations.append({
                "key": "node.kubernetes.io/unschedulable",
                "operator": "Exists",
                "effect": "NoSchedule"
            })
        
        # Build the startup command
        if github_info:
            # Download the notebook file before starting Jupyter
            notebook_filename = github_info["path"].split("/")[-1]
            startup_script = f"""
echo "{settings.PYPI_HOST_IP} {settings.PYPI_HOST}" >> /etc/hosts
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = {settings.PYPI_MIRROR}
trusted-host = {settings.PYPI_HOST}
EOF
unset PIP_EXTRA_INDEX_URL
pip install --no-cache-dir --index-url {settings.PYPI_MIRROR} --trusted-host {settings.PYPI_HOST} jupyter ihighlight
mkdir -p /app/notebooks
if [ -d "{settings.PUBLIC_MODELS_MOUNT_PATH}" ] && [ ! -e /app/models ]; then
  ln -s "{settings.PUBLIC_MODELS_MOUNT_PATH}" /app/models
fi
if [ -d "{settings.PUBLIC_MODELS_MOUNT_PATH}" ] && [ ! -e /app/notebooks/models ]; then
  ln -s "{settings.PUBLIC_MODELS_MOUNT_PATH}" /app/notebooks/models
fi
cd /app/notebooks
python -c "
import urllib.request
import ssl
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
url = '{github_info["raw_url"]}'
urllib.request.urlretrieve(url, '{notebook_filename}')
print('Downloaded: {notebook_filename}')
"
jupyter lab --ip=0.0.0.0 --port={settings.NOTEBOOK_PORT} --no-browser --allow-root --ServerApp.token='{settings.NOTEBOOK_TOKEN}' --notebook-dir=/app/notebooks
"""
        else:
            startup_script = f"""
echo "{settings.PYPI_HOST_IP} {settings.PYPI_HOST}" >> /etc/hosts
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = {settings.PYPI_MIRROR}
trusted-host = {settings.PYPI_HOST}
EOF
unset PIP_EXTRA_INDEX_URL
pip install --no-cache-dir --index-url {settings.PYPI_MIRROR} --trusted-host {settings.PYPI_HOST} jupyter ihighlight
if [ -d "{settings.PUBLIC_MODELS_MOUNT_PATH}" ] && [ ! -e /app/models ]; then
  ln -s "{settings.PUBLIC_MODELS_MOUNT_PATH}" /app/models
fi
cd /app
jupyter lab --ip=0.0.0.0 --port={settings.NOTEBOOK_PORT} --no-browser --allow-root --ServerApp.token='{settings.NOTEBOOK_TOKEN}'
"""
        
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": instance_id,
                "namespace": self.namespace,
                "labels": labels,
                "annotations": annotations
            },
            "spec": {
                **(
                    {
                        "nodeSelector": {
                            "kubernetes.io/hostname": target_node_hostname
                        }
                    }
                    if target_node_hostname else {}
                ),
                "tolerations": tolerations,
                "containers": [
                    {
                        "name": "notebook",
                        "image": image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-c"],
                        "args": [startup_script],
                        "ports": [
                            {
                                "containerPort": settings.NOTEBOOK_PORT,
                                "name": "jupyter"
                            }
                        ],
                        "resources": {
                            "limits": {
                                "cpu": settings.CPU_LIMIT,
                                "memory": settings.MEMORY_LIMIT,
                                "amd.com/gpu": settings.GPU_LIMIT
                            },
                            "requests": {
                                "cpu": settings.CPU_REQUEST,
                                "memory": settings.MEMORY_REQUEST,
                                "amd.com/gpu": settings.GPU_LIMIT
                            }
                        },
                        "env": [
                            {"name": "SHELL", "value": "/bin/bash"},
                            {"name": "USER_EMAIL", "value": email}
                        ],
                        "volumeMounts": [
                            {"name": "shm", "mountPath": "/dev/shm"},
                            {
                                "name": "public-models",
                                "mountPath": settings.PUBLIC_MODELS_MOUNT_PATH,
                                "readOnly": True
                            }
                        ]
                    }
                ],
                "volumes": [
                    {
                        "name": "shm",
                        "emptyDir": {
                            "medium": "Memory",
                            "sizeLimit": "64Gi"
                        }
                    },
                    {
                        "name": "public-models",
                        "hostPath": {
                            "path": settings.PUBLIC_MODELS_HOST_PATH,
                            "type": "DirectoryOrCreate"
                        }
                    }
                ],
                "restartPolicy": "Always"
            }
        }
    
    def _get_service_manifest(self, email: str, instance_id: str, node_port: int) -> dict:
        """Generate Service manifest"""
        labels = self._get_labels(email, instance_id)
        
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{instance_id}-svc",
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": {
                "selector": labels,
                "type": "NodePort",
                "ports": [
                    {
                        "name": "jupyter",
                        "port": settings.NOTEBOOK_PORT,
                        "targetPort": settings.NOTEBOOK_PORT,
                        "nodePort": node_port
                    }
                ]
            }
        }
    
    def _resolve_node_port_range(self, platform: Optional[str] = None) -> tuple[int, int]:
        """Resolve the allowed NodePort range for a reservation platform."""
        normalized_platform = (platform or "").strip()
        if normalized_platform:
            platform_key = _normalize_platform_key(normalized_platform)
            mapped_range = settings.NOTEBOOK_PLATFORM_PORT_RANGES.get(platform_key)
            if mapped_range:
                return mapped_range

            if settings.NOTEBOOK_PLATFORM_PORT_RANGES:
                raise ValueError(f'No notebook port range configured for platform "{normalized_platform}".')

        return settings.NODE_PORT_BASE, settings.NODE_PORT_END

    def _allocate_node_port(self, platform: Optional[str] = None) -> int:
        """Allocate an available NodePort within the platform-specific range."""
        used_ports = set()
        
        try:
            services = self.core_v1.list_namespaced_service(
                namespace=self.namespace
            )
            for svc in services.items:
                for port in svc.spec.ports or []:
                    if port.node_port:
                        used_ports.add(port.node_port)
        except ApiException as e:
            logger.warning(f"Error listing services: {e}")
        
        # Find available port within the configured range.
        start_port, end_port = self._resolve_node_port_range(platform)
        port = start_port
        while port in used_ports and port <= end_port:
            port += 1

        if port > end_port:
            raise ValueError(f"No available NodePort in configured range {start_port}-{end_port}.")
        
        return port
    
    def get_instance_by_email(self, email: str) -> Optional[dict]:
        """Get existing notebook instance for an email"""
        instance_id = self._generate_instance_id(email)
        
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=instance_id,
                namespace=self.namespace
            )
            
            # Get associated service
            try:
                svc = self.core_v1.read_namespaced_service(
                    name=f"{instance_id}-svc",
                    namespace=self.namespace
                )
                node_port = svc.spec.ports[0].node_port if svc.spec.ports else None
            except ApiException:
                node_port = None

            annotations = pod.metadata.annotations or {}
            
            return {
                "id": instance_id,
                "email": email,
                "owner_username": annotations.get("amd-oneclick/owner-username"),
                "pod_name": pod.metadata.name,
                "service_name": f"{instance_id}-svc",
                "image": pod.spec.containers[0].image,
                "status": pod.status.phase.lower(),
                "created_at": pod.metadata.creation_timestamp,
                "reservation_end_at": annotations.get("amd-oneclick/reservation-end-at"),
                "platform": annotations.get("amd-oneclick/platform"),
                "target_node_hostname": annotations.get("amd-oneclick/target-node-hostname"),
                "node_port": node_port,
                "url": self._build_url(node_port) if node_port else None
            }
        except ApiException as e:
            if e.status == 404:
                return None
            raise
    
    def _build_url(self, node_port: int, notebook_path: Optional[str] = None) -> str:
        """Build notebook URL"""
        base_url = f"http://{settings.SERVICE_HOST}:{node_port}/lab?token={settings.NOTEBOOK_TOKEN}"
        if notebook_path:
            # Add notebook path to URL for direct open
            notebook_filename = notebook_path.split("/")[-1]
            return f"http://{settings.SERVICE_HOST}:{node_port}/lab/tree/{notebook_filename}?token={settings.NOTEBOOK_TOKEN}"
        return base_url
    
    def create_instance(self, email: str, image: Optional[str] = None, 
                        github_info: Optional[dict] = None,
                        custom_instance_id: Optional[str] = None,
                        reservation_end_at: Optional[str] = None,
                        owner_username: Optional[str] = None,
                        platform: Optional[str] = None) -> dict:
        """Create a new notebook instance"""
        instance_id = custom_instance_id or self._generate_instance_id(email)
        image = image or settings.DEFAULT_IMAGE
        
        # Check if instance already exists
        existing = self.get_instance_by_id(instance_id)
        if existing:
            return existing
        
        # Allocate NodePort
        node_port = self._allocate_node_port(platform)
        
        # Create Pod
        pod_manifest = self._get_pod_manifest(
            email,
            instance_id,
            image,
            github_info,
            reservation_end_at=reservation_end_at,
            owner_username=owner_username,
            platform=platform
        )
        try:
            self.core_v1.create_namespaced_pod(
                namespace=self.namespace,
                body=pod_manifest
            )
            logger.info(f"Created pod {instance_id} for {email}")
        except ApiException as e:
            logger.error(f"Failed to create pod: {e}")
            raise
        
        # Create Service
        svc_manifest = self._get_service_manifest(email, instance_id, node_port)
        try:
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=svc_manifest
            )
            logger.info(f"Created service {instance_id}-svc with NodePort {node_port}")
        except ApiException as e:
            logger.error(f"Failed to create service: {e}")
            # Cleanup pod if service creation fails
            self.delete_instance_by_id(instance_id)
            raise
        
        notebook_path = github_info.get("path") if github_info else None
        
        return {
            "id": instance_id,
            "email": email,
            "owner_username": (owner_username or "").strip() or None,
            "pod_name": instance_id,
            "service_name": f"{instance_id}-svc",
            "image": image,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "reservation_end_at": reservation_end_at,
            "platform": (platform or "").strip() or None,
            "target_node_hostname": self._resolve_notebook_node_hostname(platform),
            "node_port": node_port,
            "url": self._build_url(node_port, notebook_path),
            "github_info": github_info
        }
    
    def get_instance_by_id(self, instance_id: str) -> Optional[dict]:
        """Get existing notebook instance by instance ID"""
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=instance_id,
                namespace=self.namespace
            )
            
            # Get associated service
            try:
                svc = self.core_v1.read_namespaced_service(
                    name=f"{instance_id}-svc",
                    namespace=self.namespace
                )
                node_port = svc.spec.ports[0].node_port if svc.spec.ports else None
            except ApiException:
                node_port = None
            
            annotations = pod.metadata.annotations or {}
            email = annotations.get("amd-oneclick/email", "unknown")
            github_path = annotations.get("amd-oneclick/github-path")
            
            return {
                "id": instance_id,
                "email": email,
                "owner_username": annotations.get("amd-oneclick/owner-username"),
                "pod_name": pod.metadata.name,
                "service_name": f"{instance_id}-svc",
                "image": pod.spec.containers[0].image,
                "status": pod.status.phase.lower(),
                "created_at": pod.metadata.creation_timestamp,
                "reservation_end_at": annotations.get("amd-oneclick/reservation-end-at"),
                "platform": annotations.get("amd-oneclick/platform"),
                "target_node_hostname": annotations.get("amd-oneclick/target-node-hostname"),
                "node_port": node_port,
                "url": self._build_url(node_port, github_path) if node_port else None,
                "github_org": annotations.get("amd-oneclick/github-org"),
                "github_repo": annotations.get("amd-oneclick/github-repo"),
                "github_path": github_path,
            }
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def update_instance_reservation_end_by_id(self, instance_id: str, reservation_end_at: str) -> bool:
        """Update reservation end timestamp annotation for an instance pod."""
        try:
            self.core_v1.patch_namespaced_pod(
                name=instance_id,
                namespace=self.namespace,
                body={
                    "metadata": {
                        "annotations": {
                            "amd-oneclick/reservation-end-at": reservation_end_at
                        }
                    }
                }
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            logger.warning("Failed to patch reservation end for %s: %s", instance_id, e)
            return False

    def update_instance_reservation_end(self, email: str, reservation_end_at: str) -> bool:
        """Update reservation end timestamp annotation by user email."""
        instance_id = self._generate_instance_id(email)
        return self.update_instance_reservation_end_by_id(instance_id, reservation_end_at)

    def update_instance_owner_username_by_id(self, instance_id: str, owner_username: str) -> bool:
        """Update owner username annotation for an instance pod."""
        normalized_owner_username = (owner_username or "").strip()
        if not normalized_owner_username:
            return False

        try:
            self.core_v1.patch_namespaced_pod(
                name=instance_id,
                namespace=self.namespace,
                body={
                    "metadata": {
                        "annotations": {
                            "amd-oneclick/owner-username": normalized_owner_username
                        }
                    }
                }
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            logger.warning("Failed to patch owner username for %s: %s", instance_id, e)
            return False

    def update_instance_owner_username(self, email: str, owner_username: str) -> bool:
        """Update owner username annotation by user email."""
        instance_id = self._generate_instance_id(email)
        return self.update_instance_owner_username_by_id(instance_id, owner_username)
    
    def delete_instance_by_id(self, instance_id: str) -> bool:
        """Delete a notebook instance by instance ID"""
        deleted = False
        
        # Delete Service
        try:
            self.core_v1.delete_namespaced_service(
                name=f"{instance_id}-svc",
                namespace=self.namespace
            )
            logger.info(f"Deleted service {instance_id}-svc")
            deleted = True
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Error deleting service: {e}")
        
        # Delete Pod
        try:
            self.core_v1.delete_namespaced_pod(
                name=instance_id,
                namespace=self.namespace
            )
            logger.info(f"Deleted pod {instance_id}")
            deleted = True
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Error deleting pod: {e}")
        
        return deleted
    
    def delete_instance(self, email: str) -> bool:
        """Delete a notebook instance"""
        instance_id = self._generate_instance_id(email)
        return self.delete_instance_by_id(instance_id)
    
    def list_instances(self) -> list:
        """List all notebook instances"""
        instances = []
        
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={settings.NOTEBOOK_LABEL_PREFIX}"
            )
            
            for pod in pods.items:
                instance_id = pod.metadata.labels.get("instance-id", "unknown")
                annotations = pod.metadata.annotations or {}
                email = annotations.get("amd-oneclick/email", "unknown")
                owner_username = annotations.get("amd-oneclick/owner-username")
                created_at = pod.metadata.creation_timestamp
                
                # Get GitHub info from annotations
                github_org = annotations.get("amd-oneclick/github-org")
                github_repo = annotations.get("amd-oneclick/github-repo")
                github_path = annotations.get("amd-oneclick/github-path")
                reservation_end_at = annotations.get("amd-oneclick/reservation-end-at")
                platform = annotations.get("amd-oneclick/platform")
                target_node_hostname = annotations.get("amd-oneclick/target-node-hostname")

                if (not owner_username) and email and email != "unknown" and "@" in email and not github_org:
                    owner_username = email.split("@", 1)[0]
                
                # Get NodePort from service
                node_port = None
                try:
                    svc = self.core_v1.read_namespaced_service(
                        name=f"{instance_id}-svc",
                        namespace=self.namespace
                    )
                    node_port = svc.spec.ports[0].node_port if svc.spec.ports else None
                except ApiException:
                    pass
                
                # Calculate uptime
                uptime_minutes = 0
                if created_at:
                    uptime_delta = datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)
                    uptime_minutes = int(uptime_delta.total_seconds() / 60)
                
                instances.append({
                    "id": instance_id,
                    "email": email,
                    "owner_username": owner_username,
                    "pod_name": pod.metadata.name,
                    "service_name": f"{instance_id}-svc",
                    "image": pod.spec.containers[0].image if pod.spec.containers else "unknown",
                    "status": pod.status.phase.lower() if pod.status.phase else "unknown",
                    "created_at": created_at.isoformat() if created_at else None,
                    "reservation_end_at": reservation_end_at,
                    "platform": platform,
                    "target_node_hostname": target_node_hostname,
                    "node_port": node_port,
                    "url": self._build_url(node_port, github_path) if node_port else None,
                    "uptime_minutes": uptime_minutes,
                    "github_org": github_org,
                    "github_repo": github_repo,
                    "github_path": github_path,
                })
        except ApiException as e:
            logger.error(f"Error listing pods: {e}")
        
        return instances
    
    def delete_all_instances(self) -> int:
        """Delete all notebook instances"""
        instances = self.list_instances()
        deleted_count = 0
        
        for instance in instances:
            if self.delete_instance(instance["email"]):
                deleted_count += 1
        
        return deleted_count
    
    def get_pod_status(self, email: str, instance_id: Optional[str] = None) -> Optional[str]:
        """Get the current status of a pod"""
        if not instance_id:
            instance_id = self._generate_instance_id(email)
        
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=instance_id,
                namespace=self.namespace
            )
            
            phase = pod.status.phase.lower() if pod.status.phase else "unknown"
            
            # Check container statuses for more detail
            if pod.status.container_statuses:
                container_status = pod.status.container_statuses[0]
                if container_status.ready:
                    # Container is ready, but we need to verify Jupyter is actually responding
                    instance = self.get_instance_by_id(instance_id)
                    if instance and instance.get("node_port"):
                        if self._check_jupyter_ready(instance["node_port"]):
                            return "ready"
                        else:
                            return "jupyter_starting"
                    return "running"
                elif container_status.state.waiting:
                    reason = container_status.state.waiting.reason or "waiting"
                    if reason in ["ContainerCreating", "PodInitializing"]:
                        return "initializing"
                    elif reason == "ImagePullBackOff":
                        return "failed"
                    return "loading"
                elif container_status.state.running:
                    # Container is running but not ready yet
                    return "running"
            
            return phase
        except ApiException as e:
            if e.status == 404:
                return None
            raise
    
    def _check_jupyter_ready(self, node_port: int, timeout: float = 2.0) -> bool:
        """Check if Jupyter is responding on the given port"""
        try:
            # Try to connect to the Jupyter server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # Connect to any node in the cluster
            result = sock.connect_ex((settings.SERVICE_HOST, node_port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Jupyter health check failed: {e}")
            return False
    
    def check_pod_activity(self, email: str) -> Optional[datetime]:
        """Check last activity of a pod by examining logs"""
        instance_id = self._generate_instance_id(email)
        
        try:
            # Get recent logs
            logs = self.core_v1.read_namespaced_pod_log(
                name=instance_id,
                namespace=self.namespace,
                tail_lines=10,
                timestamps=True
            )
            
            if logs:
                # Parse last log timestamp
                lines = logs.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    # Kubernetes log format: 2024-01-01T00:00:00.000000000Z ...
                    timestamp_str = last_line.split(' ')[0]
                    try:
                        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        pass
            
            return None
        except ApiException:
            return None
    
    def cleanup_idle_instances(self) -> list:
        """Cleanup idle and expired instances"""
        cleaned = []
        instances = self.list_instances()
        now = datetime.now(timezone.utc)
        
        for instance in instances:
            should_delete = False
            reason = ""
            reservation_end_at = self._parse_iso_datetime(instance.get("reservation_end_at"))

            if reservation_end_at and now >= reservation_end_at:
                should_delete = True
                reason = f"reservation ended at {reservation_end_at.isoformat()}"
            
            # Check max lifetime
            uptime_hours = instance["uptime_minutes"] / 60
            if not should_delete and uptime_hours >= settings.MAX_LIFETIME_HOURS:
                should_delete = True
                reason = f"exceeded max lifetime ({settings.MAX_LIFETIME_HOURS}h)"
            
            # Check idle timeout (only for running instances)
            elif not should_delete and instance["status"] == "running":
                last_activity = self.check_pod_activity(instance["email"])
                if last_activity:
                    idle_minutes = (now - last_activity).total_seconds() / 60
                    if idle_minutes >= settings.IDLE_TIMEOUT_MINUTES:
                        should_delete = True
                        reason = f"idle for {int(idle_minutes)} minutes"
            
            if should_delete:
                if self.delete_instance(instance["email"]):
                    cleaned.append({
                        "email": instance["email"],
                        "reason": reason
                    })
                    logger.info(f"Cleaned up instance for {instance['email']}: {reason}")
        
        return cleaned


# Global K8s client instance
k8s_client = K8sClient()

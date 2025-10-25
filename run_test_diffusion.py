from ast import Dict
from scripts import train_cst as cst
from core import constants as core_cst
import logging
import sys
import docker
import asyncio
import uuid
import json 
import argparse
from docker.models.containers import Container
import signal
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_wandb_env(task_id: str, hotkey: str) -> dict:
    wandb_path = f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}"

    env = {
        "WANDB_MODE": "offline",
        **{key: wandb_path for key in cst.WANDB_DIRECTORIES}
    }

    return env


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('run_test_diffusion.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


async def create_volumes_if_dont_exist():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            volume = client.volumes.get(volume_name)
        except docker.errors.NotFound:
            volume = client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


def check_cache_volume_contents():
    """Check contents of cache volume for debugging"""
    client: docker.DockerClient = docker.from_env()
    
    try:
        logger.info("Checking cache volume contents...")
        container = client.containers.run(
            image="alpine:latest",
            command=["ls", "-la", "/cache/datasets"],
            volumes={"cache": {"bind": "/cache", "mode": "ro"}},
            remove=True,
            detach=False
        )
        logger.info(f"Cache volume datasets contents: {container.decode('utf-8')}")
    except Exception as e:
        logger.warning(f"Failed to check cache volume: {e}")
        # Try to pull alpine if it doesn't exist
        try:
            logger.info("Pulling alpine image...")
            client.images.pull("alpine:latest")
            container = client.containers.run(
                image="alpine:latest", 
                command=["ls", "-la", "/cache/datasets"],
                volumes={"cache": {"bind": "/cache", "mode": "ro"}},
                remove=True,
                detach=False
            )
            logger.info(f"Cache volume datasets contents: {container.decode('utf-8')}")
        except Exception as e2:
            logger.error(f"Failed to check cache volume even after pulling alpine: {e2}")


def stream_container_logs(container: Container):
    buffer = ""
    try:
        for log_chunk in container.logs(stream=True, follow=True):
            log_text = log_chunk.decode("utf-8", errors="replace")
            buffer += log_text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line:
                    logger.info(line)
        if buffer:
            logger.info(buffer)
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")


def build_docker_image(
    dockerfile_path: str, context_path: str = ".", is_image_task: bool = True, tag: str = None, no_cache: bool = True
) -> tuple[str, Exception | None]:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:test"

    logger.info(f"Building Docker image '{tag}', Dockerfile path: {dockerfile_path}, Context Path: {context_path}...")

    try:
        # Use the high-level API instead of low-level API for better error handling
        image, build_logs = client.images.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            rm=True,
            forcerm=True
        )
        
        # Log build output
        for log in build_logs:
            if 'stream' in log:
                logger.info(log['stream'].strip())
            elif 'error' in log:
                error_msg = log['error']
                logger.error(f"Build error: {error_msg}")
                return tag, Exception(error_msg)
        
        logger.info(f"Docker image '{tag}' built successfully with ID: {image.id}")
        return tag, None
        
    except Exception as e:
        logger.error(f"Failed to build Docker image: {str(e)}")
        return tag, e


def calculate_container_resources(gpu_ids: list[int]) -> tuple[str, int]:
    """Calculate memory limit and CPU limit based on GPU count.
    
    Returns:
        tuple: (memory_limit_str, cpu_limit_nanocpus)
    """
    num_gpus = len(gpu_ids)
    memory_limit = f"{num_gpus * cst.MEMORY_PER_GPU_GB}g"
    cpu_limit_nanocpus = num_gpus * cst.CPUS_PER_GPU * 1_000_000_000
    
    logger.info(f"Allocating resources for {num_gpus} GPUs: {memory_limit} memory, {num_gpus * cst.CPUS_PER_GPU} CPUs")
    return memory_limit, cpu_limit_nanocpus


async def run_trainer_container_image(
    task_id: str,
    hotkey: str,
    tag: str,
    model: str,
    dataset_zip: str,
    model_type: str,
    expected_repo_name: str,
    hours_to_complete: float,
    gpu_ids: list[int] = [0],
) -> Container:
    client: docker.DockerClient = docker.from_env()

    # Check if image exists
    try:
        image = client.images.get(tag)
        logger.info(f"Found Docker image: {tag} (ID: {image.id[:12]})")
    except docker.errors.ImageNotFound:
        logger.error(f"Docker image '{tag}' not found. Please build the image first.")
        return Exception(f"Docker image '{tag}' not found")

    environment = build_wandb_env(task_id, hotkey)
    environment.update({
        "TRANSFORMERS_CACHE": cst.HUGGINGFACE_CACHE_PATH,
    })

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset-zip",
        dataset_zip,
        "--model-type",
        model_type,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete)
    ]

    container_name = f"image-trainer-{uuid.uuid4().hex}"
    
    # Calculate resources based on GPU count
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)
    logger.info(f"MEMORY LIMIT: {memory_limit}")
    logger.info(f"CPU LIMIT: {cpu_limit_nanocpus}")
    
    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                cst.VOLUME_NAMES[1]: {"bind": cst.CACHE_ROOT_PATH, "mode": "rw"},
            },
            remove=False,
            name=container_name,
            mem_limit=memory_limit,
            nano_cpus=cpu_limit_nanocpus,
            shm_size="8g",
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
            network_mode="none",
            environment=environment,
        )

        return container
    except Exception as e:
        logger.error(e)
        return e


def extract_container_error(logs: str) -> str | None:
    lines = logs.strip().splitlines()

    for line in reversed(lines):
        line = line.strip()
        if line and ":" in line and any(word in line for word in ["Error", "Exception"]):
            return line

    return None


def run_downloader_container(
    task_id: str,
    model: str,
    dataset_url: str,
    task_type: str = "ImageTask",
    file_format: str | None = None,
) -> tuple[int, Exception | None]:
    client = docker.from_env()

    command = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--task-type",
        task_type,
        "--dataset",
        dataset_url,
    ]
    if file_format:
        command += ["--file-format", file_format]

    container_name = f"downloader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None

    try:
        logger.info(f"Starting downloader container: {container_name}")
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            name=container_name,
            command=command,
            volumes={cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"}},
            remove=False,
            detach=True,
        )

        stream_container_logs(container)

        result = container.wait()
        exit_code = result.get("StatusCode", -1)

        if exit_code == 0:
            logger.info(f"Download completed successfully for task {task_id}")
        else:
            logs = container.logs().decode("utf-8", errors="ignore")
            error_message = extract_container_error(logs)
            return exit_code, error_message

        return exit_code, None

    except docker.errors.ContainerError as e:
        logger.error(f"Downloader container failed for task {task_id}: {e}")
        return 1, e

    except Exception as ex:
        logger.error(f"Unexpected error in downloader for task {task_id}: {ex}")
        return 1, ex

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}")


async def _force_stop_container(container: Container):
    """Aggressively stop container with multiple fallback methods"""
    try:
        # Method 1: Try graceful stop with short timeout
        container.stop(timeout=5)
        logger.info("Container stopped gracefully")
        return
    except Exception as e:
        logger.warning(f"Graceful stop failed: {e}")
    
    try:
        # Method 2: Force kill
        container.kill()
        logger.info("Container force killed")
        return  
    except Exception as e:
        logger.warning(f"Force kill failed: {e}")
    
    try:
        # Method 3: Kill with SIGKILL signal
        container.kill(signal="SIGKILL")
        logger.info("Container killed with SIGKILL")
    except Exception as e:
        logger.error(f"SIGKILL failed: {e}")


async def _handle_timeout_scenario(task: dict, container: Container) -> bool:
    """Handle timeout scenario"""
    logger.info("TIMEOUT: Starting timeout handling procedure...")
    
    # Step 1: Stop container gracefully
    try:
        logger.info("Stopping container gracefully (30s timeout)...")
        container.stop(timeout=30)
        logger.info("Container stopped gracefully")
    except Exception as e:
        logger.warning(f"Graceful stop failed: {e}")
        # Force kill if needed
        await _force_stop_container(container)
    
    # Step 2: Clean up container
    try:
        container.remove(force=True)
        logger.info("🧹 Container cleaned up after timeout")
    except Exception as cleanup_err:
        logger.warning(f"Failed to remove container: {cleanup_err}")
    
    logger.info("Timeout handling completed")
    return False


async def _handle_success_scenario(task: dict) -> bool:
    """Handle successful training"""
    logger.info("Training completed successfully!")
    return True


def run_hf_uploader_container(
    task_id: str,
    expected_repo_name: str,
    model: str,
    huggingface_token: str = None,
    huggingface_username: str = None,
) -> tuple[int, Exception | None]:
    """Run HuggingFace uploader container to upload trained model"""
    client = docker.from_env()
    
    # Use provided parameters or constants from core_cst
    hf_token = huggingface_token or core_cst.HUGGINGFACE_TOKEN
    hf_username = huggingface_username or core_cst.HUGGINGFACE_USERNAME 
    
    if not hf_token or not hf_username:
        logger.error("HUGGINGFACE_TOKEN and HUGGINGFACE_USERNAME must be provided")
        return 1, Exception("Missing HuggingFace credentials")
    
    local_folder = f"/app/checkpoints/{task_id}/{expected_repo_name}"
    
    environment = {
        "HUGGINGFACE_TOKEN": hf_token,
        "HUGGINGFACE_USERNAME": hf_username,
        "TASK_ID": task_id,
        "EXPECTED_REPO_NAME": expected_repo_name,
        "LOCAL_FOLDER": local_folder,
        "MODEL": model,
        "HF_REPO_SUBFOLDER": cst.IMAGE_TASKS_HF_SUBFOLDER_PATH
    }
    
    container_name = f"hf-uploader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None
    
    try:
        logger.info(f"Starting HuggingFace uploader container: {container_name}")
        logger.info(f"Uploading from: {local_folder}")
        logger.info(f"Target repo: {hf_username}/{expected_repo_name}")
        
        container = client.containers.run(
            image="hf-uploader",
            name=container_name,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": "/app/checkpoints", "mode": "ro"}  # checkpoints volume
            },
            environment=environment,
            remove=False,
            detach=True,
        )
        
        stream_container_logs(container)
        
        result = container.wait()
        exit_code = result.get("StatusCode", -1)
        
        if exit_code == 0:
            logger.info(f"Model uploaded successfully to HuggingFace: {hf_username}/{expected_repo_name}")
        else:
            logs = container.logs().decode("utf-8", errors="ignore")
            error_message = extract_container_error(logs)
            logger.error(f"HuggingFace upload failed with exit code {exit_code}: {error_message}")
            return exit_code, error_message
            
        return exit_code, None
        
    except docker.errors.ContainerError as e:
        logger.error(f"HuggingFace uploader container failed: {e}")
        return 1, e
    except Exception as ex:
        logger.error(f"Unexpected error in HuggingFace uploader: {ex}")
        return 1, ex
    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove uploader container {container_name}: {cleanup_err}")


async def run_trainer(task: dict, build_image: bool = False):
    logger.info(f"run_trainer function received build_image: {build_image}")
    
    timeout_seconds = int(task["hours_to_complete"] * 3600)
    model_type = task["model_type"]
    container = None
    success = False
    timeout_occurred = False
    
    logger.info(f"Training timeout set to: {timeout_seconds} seconds ({task['hours_to_complete']} hours)")
    logger.info(f"Training will be killed after: {timeout_seconds/60:.1f} minutes")
    
    await create_volumes_if_dont_exist()

    local_repo_path = "."
    dockerfile_path = cst.DEFAULT_IMAGE_DOCKERFILE_PATH

    download_status, exc = await asyncio.to_thread(
        run_downloader_container,
        task_id=task["task_id"],
        model=task["model"],
        dataset_url=task["dataset_zip"],
        task_type="ImageTask",
    )
    
    if download_status != 0:
        logger.error(f"Download failed with status {download_status}: {exc}")
        return
    
    # Check cache volume contents after download
    check_cache_volume_contents()
    
    tag = f"standalone-image-trainer:test"
    logger.info(f"Build image flag: {build_image}")
    
    if build_image:
        logger.info("Building Docker image...")
        logger.info(f"Dockerfile path: {dockerfile_path}")
        logger.info(f"Context path: {local_repo_path}")
        tag, exc = await asyncio.to_thread(
                build_docker_image,
                dockerfile_path=dockerfile_path,
                is_image_task=True,
                context_path=local_repo_path,
                tag=tag
        )
        
        if exc:
            logger.error(f"Failed to build image: {exc}")
            return
        
        logger.info(f"Successfully built Docker image: {tag}")
        
        # Also build HF uploader image
        logger.info("Building HuggingFace uploader image...")
        hf_tag, hf_exc = await asyncio.to_thread(
            build_docker_image,
            dockerfile_path="dockerfiles/hf-uploader.dockerfile",
            is_image_task=False,
            context_path=local_repo_path,
            tag="hf-uploader"
        )
        
        if hf_exc:
            logger.warning(f"Failed to build HF uploader image: {hf_exc}")
            logger.warning("HuggingFace upload will not be available")
        else:
            logger.info("Successfully built HuggingFace uploader image")
    else:
        logger.info(f"Using existing Docker image: {tag}")
    
    logger.info(f"Running task now")
    
    try:
        container = await asyncio.wait_for(
            run_trainer_container_image(
                task_id=task["task_id"],
                hotkey="ABC-TEST",
                tag=tag,
                model=task["model"],
                dataset_zip=task["dataset_zip"],
                model_type=task["model_type"],
                expected_repo_name=task["expected_repo_name"],
                hours_to_complete=task["hours_to_complete"],
                gpu_ids=task.get("gpu_ids", [0]),
            ),
            timeout=60,
        )
        
        if isinstance(container, Exception):
            logger.error(f"Failed to start container: {container}")
            return
        
            
        logger.info("Starting to stream container logs...")
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container))
        wait_task = asyncio.create_task(asyncio.to_thread(container.wait))
        
        try:
            done, pending = await asyncio.wait({wait_task}, timeout=timeout_seconds)
           
            if wait_task in done:
                result = await wait_task
                logger.info(f"Container.wait() returned: {result}")
                status_code = result.get("StatusCode", -1)
                if status_code == 0:
                    success = True
                    logger.info("Training completed successfully")
                else:
                    success = False
                    logger.warning(f"Training failed with status code: {status_code}")
            else:
                logger.error(f"TIMEOUT REACHED ({timeout_seconds}s / {task['hours_to_complete']} hours)")
                logger.error("Starting timeout handling procedure...")
                
                # Cancel both wait and log tasks immediately
                wait_task.cancel()
                log_task.cancel()
                
                # Handle timeout scenario (includes container stop + upload)
                timeout_success = await _handle_timeout_scenario(task, container)
                success = timeout_success
                timeout_occurred = True
                
        except Exception as e:
            logger.error(f"Error during container execution: {e}")
            success = False
            
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for container to start")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        
        # Log final results
        if success and not timeout_occurred:
            logger.info("Training completed successfully!")
            
            # Upload to HuggingFace if training was successful
            logger.info("Starting HuggingFace upload...")
            upload_status, upload_exc = await asyncio.to_thread(
                run_hf_uploader_container,
                task_id=task["task_id"],
                expected_repo_name=task["expected_repo_name"],
                model=task["model"],
                huggingface_token=task.get("hf_token"),
                huggingface_username=task.get("hf_username"),
            )
            
            if upload_status == 0:
                logger.info("Model uploaded to HuggingFace successfully!")
            else:
                logger.warning(f"HuggingFace upload failed: {upload_exc}")
                
        elif timeout_occurred:
            if success:
                logger.info("Timeout handled successfully!")
            else:
                logger.warning("Timeout occurred")
        else:
            logger.info("Training failed")
        
        # Clean up container (only if not already handled by timeout)
        if container and not isinstance(container, Exception) and not timeout_occurred:
            try:
                container.remove(force=True)
                logger.info("🧹 Container cleaned up")
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container: {cleanup_err}")
        elif timeout_occurred:
            logger.info("Container cleanup handled by timeout procedure")


def main():
    parser = argparse.ArgumentParser(description="Run diffusion training task")
    parser.add_argument("task_path", help="Path to the task JSON file")
    parser.add_argument("--build-image", action="store_true", help="Build Docker image before running")
    parser.add_argument("--hf-token", help="HuggingFace token for uploading models")
    parser.add_argument("--hf-username", help="HuggingFace username for uploading models")
    
    args = parser.parse_args()
    
    logger.info(f"Main function received build_image: {args.build_image}")
    
    with open(args.task_path, "r") as f:
        task = json.load(f)
    
    # Add HF credentials to task if provided via command line
    if args.hf_token:
        task["hf_token"] = args.hf_token
    if args.hf_username:
        task["hf_username"] = args.hf_username
    
    # Show task configuration
    logger.info("=" * 60)
    logger.info("TASK CONFIGURATION")
    logger.info("=" * 60)
    
    hours_to_complete = task.get("hours_to_complete", 0)
    logger.info(f"Training duration: {hours_to_complete} hours")
    logger.info(f"Task ID: {task.get('task_id', 'N/A')}")
    logger.info(f"Model: {task.get('model', 'N/A')}")
    logger.info(f"Dataset ZIP: {task.get('dataset_zip', 'N/A')}")
    logger.info(f"Model Type: {task.get('model_type', 'N/A')}")
    logger.info(f"Expected Repo Name: {task.get('expected_repo_name', 'N/A')}")
    
    # Show HF credentials status
    hf_token = task.get("hf_token") or core_cst.HUGGINGFACE_TOKEN
    hf_username = task.get("hf_username") or core_cst.HUGGINGFACE_USERNAME
    logger.info(f"HuggingFace Username: {hf_username or 'Not provided'}")
    logger.info(f"HuggingFace Token: {'***' if hf_token else 'Not provided'}")
    logger.info("=" * 60)
    
    asyncio.run(run_trainer(task, args.build_image))


if __name__ == "__main__":
   main()

# Scalable 4D Perception Engine for Autonomous Systems

**A high-throughput distributed ETL pipeline for multi-modal sensor data processing.**

---

### Context
Autonomous vehicles generate **4TB+ of multi-modal sensor data daily** (LiDAR point clouds, camera streams, IMU telemetry), but **95% is redundant**—only rare edge cases (occlusions, adverse weather, sensor misalignment) improve model robustness.

### Challenge
Existing preprocessing pipelines couldn't filter signal from noise at scale. ML teams waited **3–5 days** to curate training datasets, blocking rapid iteration on perception models critical for safe navigation.

### Solution Built
*   **Architected a distributed ETL pipeline** processing **50TB+ weekly** across **200+ EC2 spot instances**, orchestrated via **Kubernetes** with auto-scaling based on S3 event triggers.
*   **Developed GPU-accelerated coordinate transformation kernels** (custom CUDA/C++ bindings) for LiDAR-to-camera projection, achieving **12x speedup** over CPU-based numpy operations.
*   **Implemented intelligent data sampling using PyTorch embeddings**: trained a lightweight classifier to surface "hard examples" (night-time occlusions, construction zones, sensor noise) with **89% precision**, reducing dataset size by **60%** while maintaining model accuracy.
*   **Built interactive visualization layer with Voxel51**, enabling engineers to query 3D scenes semantically ("show me all cyclists in rain within 50m").

### Impact
*   **40% faster data retrieval** → Compressed ML iteration cycles from **4 days to 18 hours**.
*   **$23K/month AWS cost savings** through spot instance optimization and reduced redundant storage.
*   **Enabled 2.3x more training experiments in Q4**, directly contributing to a **15% improvement in object detection mAP** for edge cases.

### Technical Depth
*   Applied **homogeneous transformations (4×4 matrices)** for sensor fusion across moving reference frames.
*   Parallelized **PySpark jobs with custom partitioners** to avoid shuffle bottlenecks on **10B+ point clouds**.
*   Containerized pipeline with **Docker multi-stage builds**, reducing image size by **68%** for faster deployments.

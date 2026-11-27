# StrataDocs AI — Cloud-Native Multimodal RAG & Agentic Document Workspace on AWS

## AWS Architecture

**StrataDocs AI** is a stateless, self-healing SaaS document intelligence platform deployed on **Amazon Web Services (`ap-south-1`)**. It ingests native PDFs, scanned documents, handwritten notes (images), and Word files (`.docx`), indexes 768-dimensional vector embeddings into **PostgreSQL 16 (`pgvector`) on Amazon EBS**, stores raw and generated artifacts privately in **Amazon S3** via **IAM Instance Profiles**, and serves real-time semantic Q&A and downloadable CSV/PDF reports behind an **Application Load Balancer (ALB)** and **Auto Scaling Group (ASG)**.

---

## Architecture Highlights

* **2-Tier Decoupled Cloud Design:** Stateless FastAPI compute nodes (`t3.micro`) scale horizontally behind `DocWorkspace-ALB` while state resides durably in Amazon S3 and EBS-backed PostgreSQL (`10.0.1.95`).
* **NAT-Less Cost-Optimized VPC:** Multi-AZ public subnets with subnet-level `Auto-assign public IPv4` enable direct outbound HTTPS egress to Google Gemini API and Amazon S3 via the Internet Gateway without a managed NAT Gateway.
* **Defense-in-Depth Security:** Chained Security Groups (`ALB-SG :80` -> `App-EC2-SG :8000` -> `DB-SG :5432`), zero hardcoded AWS credentials via `DocWorkspace-EC2-Role`, and 5-minute cryptographic S3 Pre-Signed URLs.
* **Immutable Infrastructure:** `DocWorkspace-Golden-AMI-v2` pre-packages the Python `.venv` and `systemd` service (`deploy/docworkspace.service`), enabling ~45-second zero-config node replacement and CPU-based scale-out.
* **Sub-3s Multimodal RAG & Exports:** Powered by `gemini-3.5-flash-lite` (`thinking_budget=0` + 22s hard timeout) and `gemini-embedding-001` (`768d`), featuring adaptive chunking for 200+ page PDFs with live NDJSON progress streaming.

---

## Repository Structure

```text
stratadocs-ai-aws/
├── main.py                  # FastAPI backend: S3 IAM, PyMuPDF OCR, pgvector RAG, Gemini Flash-Lite
├── index.html               # SaaS web interface: Tailwind CSS, NDJSON streaming, Marked.js
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Environment variable template (copy to .env on EC2)
├── .gitignore               # Excludes *.pem, .env, envfile.txt, and .venv/ from Git
├── deploy/
│   ├── docworkspace.service # systemd daemon unit for automatic boot startup & crash recovery
│   └── docker-pgvector.sh   # PostgreSQL 16 + pgvector Docker container setup on EBS
├── docs/
│   ├── aws_architecture.png # 2-Tier AWS Architecture Diagram
│   ├── StrataDocs_AI_Final_Report-v2.docx
│   └── StrataDocs_AI_Presentation-v2.pptx
└── README.md
```

---

## Deployment Quickstart

### 1. Database Tier (`DocWorkspace-DB`)
Attach a persistent Amazon EBS volume at `/mnt/pgdata`, configure `DB_PASSWORD`, and run:
```bash
chmod +x deploy/docker-pgvector.sh
./deploy/docker-pgvector.sh
```

### 2. Application Tier (`DocWorkspace-App` / Golden AMI)
```bash
mkdir -p ~/app && cd ~/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # Populate GEMINI_API_KEY, S3_BUCKET_NAME, and DB_HOST
sudo cp deploy/docworkspace.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now docworkspace
```

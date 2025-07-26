# 🌺 Kubeflow Pipelines on Minikube – Setup, Troubleshooting & Iris Classification Demo 🌱

## 🚀 Project Overview

Welcome! This guide walks you through setting up **Kubeflow Pipelines (KFP)** on a local **Minikube** Kubernetes cluster and demonstrates an end-to-end ML workflow using the classic **Iris Flower Classification** problem.

It includes step-by-step setup, pipeline authoring/running, common troubleshooting, and deploying Kubeflow Pipelines manifests.

## 💻 Prerequisites

- **OS:** macOS, Linux, or Windows 10+
- [Minikube](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker Engine](https://docs.docker.com/get-docker/) (or containerd)
- **Python 3.7+**
- Recommended: Python virtual environment tool (e.g., `venv`)

## ⚡️ Minikube & Kubeflow Pipelines: Setup Guide

### 1️⃣ Start Minikube with Adequate Resources

```bash
minikube start 
```

*If you face startup issues, try switching the container runtime:*

```bash
minikube start --container-runtime=containerd
```

### 2️⃣ Install Kubeflow Pipelines SDK (`kfp`)

Before deploying Kubeflow Pipelines manifests, you should install the Kubeflow Pipelines Python SDK and CLI tool to interact with the pipeline server later:

1. **Create and activate a Python virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Upgrade pip and install `kfp`:**

```bash
pip install --upgrade pip
pip install kfp
```

3. **Ensure `kfp` CLI is accessible:**

If you installed with `--user`, add `~/.local/bin` to your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

4. **Verify installation:**

```bash
kfp --help
```

If the command is found, you are ready to use `kfp` CLI commands.

### 3️⃣ Deploying Kubeflow Pipelines

The installation process for Kubeflow Pipelines is the same for all three environments covered in this guide: kind, K3s, Docker-desktop, and K3ai.

To deploy Kubeflow Pipelines, run the following commands:

```bash
export PIPELINE_VERSION=2.4.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

> The Kubeflow Pipelines deployment may take several minutes to complete.

### 4️⃣ Verify Kubeflow Pipelines UI Access

Use port forwarding to access the Kubeflow Pipelines UI locally:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Open your browser and go to:

[http://localhost:8080](http://localhost:8080)

## 🏵️ Iris Flower Classification Demo with Kubeflow

### Example `kubeflow_demo.py` Structure

```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

def load_data(output_csv: str):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv(output_csv, index=False)

def train_model(input_csv: str, output_model: str):
    df = pd.read_csv(input_csv)
    X, y = df.iloc[:, :-1], df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy: {acc}")
    joblib.dump(clf, output_model)

load_data_op = create_component_from_func(load_data, packages_to_install=['scikit-learn', 'pandas'])
train_model_op = create_component_from_func(train_model, packages_to_install=['scikit-learn', 'pandas', 'joblib'])

@dsl.pipeline(
    name='Iris Classification Pipeline',
    description='Trains and evaluates a RandomForest on the Iris dataset'
)
def iris_pipeline():
    data = load_data_op('/tmp/iris.csv')
    train_model_op(data.output, '/tmp/model.joblib')

if __name__ == '__main__':
    client = kfp.Client()  # add host if needed
    kfp.compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')
    client.create_run_from_pipeline_func(iris_pipeline, arguments={})
```

### How to Run

1. Activate your virtual environment, if not already active:

```bash
source .venv/bin/activate
```

2. Run your demo script:

```bash
python kubeflow_demo.py
```

3. Create the pipeline in Kubeflow Pipelines UI using the CLI:

```bash
kfp pipeline create -p kubeflow_pipeline.yaml
```

> This command uploads the pipeline definition and registers it with the Kubeflow Pipelines backend.

4. Open the Kubeflow Pipelines UI at:

[http://localhost:8080](http://localhost:8080)

5. From the UI, locate your uploaded pipeline and **manually trigger runs** to execute the pipeline and view the outputs.

## 🧰 Troubleshooting Cheat Sheet

| 🚩 Issue                                            | 🩹 Solution                                                         |
|-----------------------------------------------------|--------------------------------------------------------------------|
| API server never starts                              | Increase Minikube CPUs and memory; restart cluster                 |
| Pods in CrashLoopBackOff or not Ready               | Check pod logs; fix missing dependencies/resource constraints      |
| `kfp: command not found`                             | Install `kfp` in virtualenv; add `~/.local/bin` to PATH            |
| 503 "no endpoints available for service"            | Backend down—check pods status, restart or fix pods                |
| `kubectl` TLS handshake timeout                      | Restart Minikube; check API server health & proxies/firewall       |
| Python LibreSSL/OpenSSL warning                      | Usually safe to ignore unless causing errors; consider Python upgrade |

## 📚 Further Reading

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
- [Official Guide: Deploying Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/localcluster-deployment/#deploying-kubeflow-pipelines)
- [Iris Dataset ML Example - scikit-learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/localcluster-deployment/

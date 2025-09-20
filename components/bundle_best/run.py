import argparse, os, json, shutil
import mlflow, mlflow.pyfunc
import requests
from itertools import islice

def _chunks(lst, n=10):
    it = iter(lst)
    for first in it:
        yield [first] + list(islice(it, n-1))

def _call_azure_sentiment(texts, endpoint, key, lang="es", api_version="2023-04-01", timeout=5.0):
    """
    Soporta:
      - Text Analytics v3.x  -> api_version como 'v3.1' o 'v3.2'
      - Analyze Text 2023+   -> api_version como '2023-04-01' (recomendado)
    Devuelve una lista con los scores por texto.
    """
    base = endpoint.rstrip('/')
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}

    outputs = []
    docs = [{"id": str(i), "language": lang, "text": t} for i, t in enumerate(texts)]

    if api_version.startswith("v"):  # ---- Text Analytics v3.x
        url = f"{base}/text/analytics/{api_version}/sentiment"
        for batch in _chunks(docs, 10):
            resp = requests.post(url, headers=headers, json={"documents": batch}, timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f"{resp.status_code} {resp.text[:300]}")
            data = resp.json()
            by_id = {d["id"]: d for d in data.get("documents", [])}
            for d in batch:
                got = by_id.get(d["id"])
                if got:
                    cs = got.get("confidenceScores", {})
                    outputs.append({
                        "azure_sentiment": got.get("sentiment"),
                        "azure_positive": cs.get("positive"),
                        "azure_neutral":  cs.get("neutral"),
                        "azure_negative": cs.get("negative"),
                    })
                else:
                    outputs.append({"azure_sentiment": None, "azure_positive": None, "azure_neutral": None, "azure_negative": None})
        return outputs

    else:  # ---- Analyze Text (Language) 2023-04-01+
        url = f"{base}/language/:analyze-text?api-version={api_version}"
        for batch in _chunks(docs, 10):
            payload = {
                "kind": "SentimentAnalysis",
                "analysisInput": {"documents": batch},
                "parameters": {"opinionMining": False}
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f"{resp.status_code} {resp.text[:300]}")
            data = resp.json()
            # La forma nueva trae los docs en data["results"]["documents"]
            docs_out = (data.get("results") or {}).get("documents", [])
            by_id = {d["id"]: d for d in docs_out}
            for d in batch:
                got = by_id.get(d["id"])
                if got:
                    cs = got.get("confidenceScores", {})
                    outputs.append({
                        "azure_sentiment": got.get("sentiment"),
                        "azure_positive": cs.get("positive"),
                        "azure_neutral":  cs.get("neutral"),
                        "azure_negative": cs.get("negative"),
                    })
                else:
                    outputs.append({"azure_sentiment": None, "azure_positive": None, "azure_neutral": None, "azure_negative": None})
        return outputs


def ensure_dir(p): os.makedirs(p, exist_ok=True)

class TfIdfSentiment(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["vectorizer"], "rb") as f:
            self.vec = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        import pandas as pd, numpy as np, os

        if "text" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'text' column.")
        texts = model_input["text"].astype(str).fillna("")
        X = self.vec.transform(texts.tolist())

        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[:, 1]
        else:
            margin = self.model.decision_function(X)
            prob = 1.0 / (1.0 + np.exp(-margin))

        df_local = pd.DataFrame({
            "pred_prob": prob,
            "pred_label": (prob >= 0.5).astype(int),
        })

        ep  = os.getenv("COGSENT_ENDPOINT")
        key = os.getenv("COGSENT_KEY")
        if ep and key:
            try:
                lang = os.getenv("COGSENT_LANG", "en")   # usa "es" si tu dominio es español
                api  = os.getenv("COGSENT_API", "v3.1")
                scores = _call_azure_sentiment(texts.tolist(), ep, key, lang=lang, api_version=api)
                df_az = pd.DataFrame(scores)
                df_az["azure_label"] = (df_az["azure_sentiment"] == "positive").astype("Int64")
                return pd.concat([df_local, df_az], axis=1)
            except Exception as e:
                df_local["azure_error"] = str(e)[:200]
                return df_local

        return df_local


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection_json", required=True)
    ap.add_argument("--model_lr", required=True)
    ap.add_argument("--model_svm", required=True)
    ap.add_argument("--vectorizer_pkl", required=True)
    ap.add_argument("--best_bundle", required=True)
    ap.add_argument("--mlflow_model", required=True)
    args = ap.parse_args()

    # 1) leer ganador
    winner = json.load(open(args.selection_json))["winner"]  # "lr" | "svm"
    src_model = args.model_svm if winner == "svm" else args.model_lr

    # 2) empaquetar bundle clásico
    ensure_dir(args.best_bundle)
    shutil.copyfile(src_model,            os.path.join(args.best_bundle, "model.pkl"))
    shutil.copyfile(args.vectorizer_pkl,  os.path.join(args.best_bundle, "vectorizer.pkl"))
    with open(os.path.join(args.best_bundle, "model_spec.json"), "w") as f:
        json.dump({"winner": winner}, f, indent=2)

    # 3) construir modelo MLflow (pyfunc)
    conda_env = {
    "name": "tfidf-sentiment-infer",
    "channels": ["conda-forge", "defaults"],
    "dependencies": [
        "python=3.10",
        "pip",                         
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas",
        "scikit-learn>=1.3",
        {"pip": [
        "mlflow>=2.5.0",
        "cloudpickle==3.1.1",        
        "ruamel.yaml==0.17.21",      
        "azureml-inference-server-http",
        "requests"
        ]}
    ]
    }


    import pandas as pd
    input_example = pd.DataFrame({"text": ["great food", "terrible service"]})

    ensure_dir(args.mlflow_model)
    mlflow.pyfunc.save_model(
        path=args.mlflow_model,
        python_model=TfIdfSentiment(),
        artifacts={"vectorizer": args.vectorizer_pkl, "model": src_model},
        conda_env=conda_env,
        input_example=input_example
        # signature=mlflow.models.infer_signature(input_example, pd.DataFrame({"pred_prob":[0.9], "pred_label":[1]}))
    )

    print(f"[bundle_best] winner={winner} | bundle -> {args.best_bundle} | mlflow_model -> {args.mlflow_model}")

    mlflow.log_artifacts(args.best_bundle, artifact_path="best_bundle")
    mlflow.log_artifacts(args.mlflow_model, artifact_path="mlflow_model_artifact")

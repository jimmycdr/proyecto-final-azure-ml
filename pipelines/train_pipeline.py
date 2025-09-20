from azure.ai.ml import dsl, Input, Output, load_component
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input

ingest          = load_component(source="../components/ingest/component.yml")
select_cols     = load_component(source="../components/select_cols/component.yml")
drop_empty_dupes= load_component(source="../components/drop_empty_dupes/component.yml")
make_label      = load_component(source="../components/make_label/component.yml")
clean_text      = load_component(source="../components/clean_text/component.yml")
split_strat     = load_component(source="../components/split_stratified/component.yml")
tfidf_fit       = load_component(source="../components/tfidf_fit/component.yml")
tfidf_transform = load_component(source="../components/tfidf_transform/component.yml")

train_lr        = load_component(source="../components/train_lr/component.yml")
train_svm       = load_component(source="../components/train_svm/component.yml")

compare_models  = load_component(source="../components/compare_models/component.yml")

COMPUTE_NAME = "cpu-cluster"

@dsl.pipeline(compute=COMPUTE_NAME, description="Amazon Reviews - Training Full + Cognitive + Registry")
def amazon_train_pipeline(reviews_csv: Input(type="uri_file"),
                          cog_endpoint: str = "",
                          use_cognitive: bool = False,
                          registered_model_name: str = "amazon-sentiment-model"):
    # Data prep
    ing = ingest(input_csv=reviews_csv)
    sel = select_cols(input_parquet=ing.outputs.output_parquet)
    ded = drop_empty_dupes(input_parquet=sel.outputs.output_parquet)
    lab = make_label(input_parquet=ded.outputs.output_parquet)
    cln = clean_text(input_parquet=lab.outputs.output_parquet)

    # Split
    spl = split_strat(input_parquet=cln.outputs.output_parquet,
                      label_col="label",
                      test_size=0.2,
                      random_state=42)

    # TF-IDF
    fit = tfidf_fit(train_parquet=spl.outputs.output_train,
                    text_col="text_clean",
                    max_features=30000,
                    ngram_range="1,2",
                    min_df=3)
    trf = tfidf_transform(train_parquet=spl.outputs.output_train,
                          test_parquet=spl.outputs.output_test,
                          vectorizer_pkl=fit.outputs.output_vectorizer,
                          text_col="text_clean",
                          label_col="label")

    lr = train_lr(
        X_train_npz=trf.outputs.output_X_train,
        y_train_parquet=trf.outputs.output_y_train,
        X_test_npz=trf.outputs.output_X_test,
        y_test_parquet=trf.outputs.output_y_test,
        y_col="label",
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        l1_ratio=0.0,
    )

    svm = train_svm(
        X_train_npz=trf.outputs.output_X_train,
        y_train_parquet=trf.outputs.output_y_train,
        X_test_npz=trf.outputs.output_X_test,
        y_test_parquet=trf.outputs.output_y_test,
    )

    cmp = compare_models(
        metrics_a_json=lr.outputs.output_metrics_json,
        metrics_b_json=svm.outputs.output_metrics_json,
        name_a="lr", name_b="svm",
    )
    # Pipeline outputs
    return {
        "vectorizer":    fit.outputs.output_vectorizer,
        "model_lr":      lr.outputs.output_model_pkl,
        "metrics_lr":    lr.outputs.output_metrics_json,
        "model_svm":     svm.outputs.output_model_pkl,
        "metrics_svm":   svm.outputs.output_metrics_json,
        "selection":     cmp.outputs.output_selection_json,
        # "pred_test_lr":  pred_test_lr.outputs.output_predictions_parquet,
        # "pred_test_svm": pred_test_svm.outputs.output_predictions_parquet
    }



if __name__ == "__main__":
    SUBSCRIPTION_ID = '251bc487-28ea-45dc-885b-101f218b8cf0'
    RESOURCE_GROUP  = "rg-ml-1"
    WORKSPACE_NAME  = "ws-ml-2"

    REVIEWS_CSV = "../data/Reviews.csv"

    # Cognitive (optional)
    USE_COGNITIVE = False
    COG_ENDPOINT  = ""

    REGISTERED_MODEL_NAME = "amazon-sentiment-model"
    EXPERIMENT_NAME = "amazon-reviews-train"

    cred = DefaultAzureCredential()
    ml_client = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    pipe = amazon_train_pipeline(
        reviews_csv=Input(type="uri_file", path=REVIEWS_CSV),
        cog_endpoint=COG_ENDPOINT,
        use_cognitive=USE_COGNITIVE,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    job = ml_client.jobs.create_or_update(pipe, experiment_name=EXPERIMENT_NAME)
    print("[submit_train] Submitted. Job name:", job.name)
    print("Studio URL:", job.studio_url)
import joblib
from pathlib import Path

ensemble = joblib.load("ensemble_models.pkl")
print(f"Tipo: {type^(ensemble^)}")

output_dir = Path("individual_models")
output_dir.mkdir(exist_ok=True)

if hasattr(ensemble, 'estimators_'):
    print("Extrayendo modelos...")
    for name, model in ensemble.named_estimators_.items():
        path = output_dir / f"{name}.pkl"
        joblib.dump(model, path)
        size = path.stat().st_size / (1024**2)
        print(f"? {name}: {size:.2f} MB")
elif isinstance(ensemble, dict):
    print("Extrayendo del diccionario...")
    for key, value in ensemble.items():
        if hasattr(value, 'predict'):
            path = output_dir / f"{key}.pkl"
            joblib.dump(value, path)
            size = path.stat().st_size / (1024**2)
            print(f"? {key}: {size:.2f} MB")

print(f"Listo! Modelos en: {output_dir}")

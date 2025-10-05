"""
Exoplanet Prediction Module
Make predictions on new transit data
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def load_trained_model(model_dir="models"):
    """Load the trained model and scaler"""
    model_path = os.path.join(model_dir, "best_model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    print("Loading trained model and scaler...")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("✓ Model and scaler loaded successfully")

    return model, scaler


def predict_single_exoplanet(model, scaler, features_dict):
    """
    Predict if a single transit signal is an exoplanet

    Args:
        model: Trained Keras model
        scaler: Fitted scaler
        features_dict: Dictionary with feature values

    Returns:
        prediction: 'Exoplanet' or 'Non-Exoplanet'
        confidence: Probability score (0-1)
    """
    # Feature names in order
    feature_names = [
        "tce_period",
        "tce_time0bk",
        "tce_duration",
        "tce_depth",
        "tce_prad",
        "tce_sma",
        "tce_impact",
        "tce_model_snr",
    ]

    # Extract features in correct order
    features = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Reshape for CNN
    features_cnn = features_scaled.reshape(1, 8, 1)

    # Predict
    probability = model.predict(features_cnn, verbose=0)[0][0]
    prediction = "Exoplanet" if probability > 0.5 else "Non-Exoplanet"
    confidence = probability if probability > 0.5 else 1 - probability

    return prediction, probability, confidence


def predict_batch(model, scaler, data_df):
    """
    Predict for multiple transit signals

    Args:
        model: Trained Keras model
        scaler: Fitted scaler
        data_df: DataFrame with feature columns

    Returns:
        DataFrame with predictions and probabilities
    """
    feature_names = [
        "tce_period",
        "tce_time0bk",
        "tce_duration",
        "tce_depth",
        "tce_prad",
        "tce_sma",
        "tce_impact",
        "tce_model_snr",
    ]

    # Extract and scale features
    X = data_df[feature_names].values
    X_scaled = scaler.transform(X)
    X_cnn = X_scaled.reshape(X_scaled.shape[0], 8, 1)

    # Predict
    probabilities = model.predict(X_cnn, verbose=0).flatten()
    predictions = ["Exoplanet" if p > 0.5 else "Non-Exoplanet" for p in probabilities]
    confidences = [p if p > 0.5 else 1 - p for p in probabilities]

    # Create results DataFrame
    results = data_df.copy()
    results["prediction"] = predictions
    results["probability"] = probabilities
    results["confidence"] = confidences

    return results


def create_sample_data():
    """Create sample transit data for testing"""
    samples = [
        {
            "name": "Hot Jupiter Candidate",
            "tce_period": 3.5,
            "tce_time0bk": 135.2,
            "tce_duration": 4.5,
            "tce_depth": 850.0,
            "tce_prad": 3.2,
            "tce_sma": 0.08,
            "tce_impact": 0.3,
            "tce_model_snr": 22.5,
        },
        {
            "name": "Super-Earth Candidate",
            "tce_period": 25.3,
            "tce_time0bk": 138.7,
            "tce_duration": 6.8,
            "tce_depth": 320.0,
            "tce_prad": 1.6,
            "tce_sma": 0.18,
            "tce_impact": 0.25,
            "tce_model_snr": 18.2,
        },
        {
            "name": "False Positive (Binary Star)",
            "tce_period": 1.8,
            "tce_time0bk": 132.1,
            "tce_duration": 1.2,
            "tce_depth": 1500.0,
            "tce_prad": 0.8,
            "tce_sma": 0.05,
            "tce_impact": 0.9,
            "tce_model_snr": 12.1,
        },
        {
            "name": "Earth-like Candidate",
            "tce_period": 365.25,
            "tce_time0bk": 140.5,
            "tce_duration": 13.0,
            "tce_depth": 84.0,
            "tce_prad": 1.0,
            "tce_sma": 1.0,
            "tce_impact": 0.2,
            "tce_model_snr": 15.8,
        },
        {
            "name": "Noise/Artifact",
            "tce_period": 0.8,
            "tce_time0bk": 133.9,
            "tce_duration": 0.5,
            "tce_depth": 2200.0,
            "tce_prad": 0.5,
            "tce_sma": 0.02,
            "tce_impact": 1.1,
            "tce_model_snr": 8.5,
        },
    ]

    return samples


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================


def run_interactive_demo():
    """Run an interactive prediction demo"""
    print("\n" + "=" * 70)
    print("EXOPLANET PREDICTION DEMO")
    print("=" * 70)

    # Load model
    model, scaler = load_trained_model()

    # Get sample data
    samples = create_sample_data()

    print("\nPredicting for sample transit signals...\n")

    for i, sample in enumerate(samples, 1):
        name = sample.pop("name")
        prediction, probability, confidence = predict_single_exoplanet(
            model, scaler, sample
        )

        print(f"Sample {i}: {name}")
        print(f"  Prediction: {prediction}")
        print(f"  Probability: {probability:.4f}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print()

    print("=" * 70)


def predict_from_csv(csv_path, output_path=None):
    """
    Predict exoplanets from a CSV file

    Args:
        csv_path: Path to CSV file with transit data
        output_path: Path to save predictions (optional)

    Returns:
        DataFrame with predictions
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("Loading model...")
    model, scaler = load_trained_model()

    print("Making predictions...")
    results = predict_batch(model, scaler, df)

    # Display summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(results)}")
    print(f"Predicted Exoplanets: {(results['prediction'] == 'Exoplanet').sum()}")
    print(
        f"Predicted Non-Exoplanets: {(results['prediction'] == 'Non-Exoplanet').sum()}"
    )
    print(f"Average confidence: {results['confidence'].mean()*100:.2f}%")

    # Save if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")

    return results


def predict_custom_input():
    """Predict with custom user input"""
    print("\n" + "=" * 70)
    print("CUSTOM EXOPLANET PREDICTION")
    print("=" * 70)
    print("\nEnter transit parameters:")

    try:
        features = {
            "tce_period": float(input("Orbital Period (days): ")),
            "tce_time0bk": float(input("Transit Epoch (BKJD): ")),
            "tce_duration": float(input("Transit Duration (hours): ")),
            "tce_depth": float(input("Transit Depth (ppm): ")),
            "tce_prad": float(input("Planetary Radius (Earth radii): ")),
            "tce_sma": float(input("Semi-major Axis (AU): ")),
            "tce_impact": float(input("Impact Parameter (0-1): ")),
            "tce_model_snr": float(input("Signal-to-Noise Ratio: ")),
        }

        model, scaler = load_trained_model()
        prediction, probability, confidence = predict_single_exoplanet(
            model, scaler, features
        )

        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"Classification: {prediction}")
        print(f"Probability: {probability:.4f}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure all inputs are valid numbers.")


# ============================================================================
# MAIN MENU
# ============================================================================


def main():
    """Main function with menu options"""
    print("\n" + "=" * 70)
    print("EXOPLANET DETECTION - PREDICTION MODULE")
    print("=" * 70)
    print("\nOptions:")
    print("1. Run demo with sample data")
    print("2. Predict from CSV file")
    print("3. Enter custom transit parameters")
    print("4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        run_interactive_demo()
    elif choice == "2":
        csv_path = input("Enter CSV file path: ").strip()
        output_path = input("Enter output path (or press Enter to skip): ").strip()
        output_path = output_path if output_path else None
        predict_from_csv(csv_path, output_path)
    elif choice == "3":
        predict_custom_input()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid option. Please run again.")


if __name__ == "__main__":
    # For quick demo, just run:
    run_interactive_demo()

    # For full menu:
    # main()

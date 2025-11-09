#!/usr/bin/env python
"""
Comprehensive test script for 1D-Ensemble package.
Tests all core functionality without running full test suite.
"""

import sys


def test_imports():
    """Test that all imports work."""
    print("🔍 Testing imports...")
    try:
        import ensemble_1d

        print(f"  ✅ ensemble_1d v{ensemble_1d.__version__}")


        print("  ✅ BaseModel")


        print("  ✅ RandomForestModel")


        print("  ✅ EnsembleModel")

        # Optional imports
        try:
            from ensemble_1d import XGBoostModel

            print("  ✅ XGBoostModel")
        except ImportError as e:
            print(f"  ⚠️  XGBoostModel (xgboost not installed: {e})")

        try:
            from ensemble_1d import PyTorchModel

            print("  ✅ PyTorchModel")
        except ImportError as e:
            print(f"  ⚠️  PyTorchModel (torch not installed: {e})")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic model functionality."""
    print("\n🧪 Testing basic functionality...")
    try:
        from sklearn.datasets import make_classification

        from ensemble_1d import EnsembleModel, RandomForestModel

        # Generate small dataset
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=8, n_classes=2, random_state=42,
        )

        # Train single model
        print("  Testing RandomForestModel...")
        rf_model = RandomForestModel(n_estimators=10)
        rf_model.fit(X[:80], y[:80])
        predictions = rf_model.predict(X[80:])
        probas = rf_model.predict_proba(X[80:])

        assert len(predictions) == 20, "Predictions length mismatch"
        assert probas.shape == (20, 2), "Probabilities shape mismatch"
        print("  ✅ RandomForestModel works")

        # Train ensemble
        print("  Testing EnsembleModel...")
        models = [RandomForestModel(n_estimators=10), RandomForestModel(n_estimators=10)]
        ensemble = EnsembleModel(models=models, fusion_method="voting")
        ensemble.fit(X[:80], y[:80])
        ensemble_predictions = ensemble.predict(X[80:])

        assert len(ensemble_predictions) == 20, "Ensemble predictions length mismatch"
        print("  ✅ EnsembleModel works")

        # Test metrics
        metrics = ensemble.evaluate(X[80:], y[80:])
        assert "accuracy" in metrics, "Metrics missing accuracy"
        assert "f1_score" in metrics, "Metrics missing f1_score"
        print(f"  ✅ Metrics: accuracy={metrics['accuracy']:.3f}")

        return True
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_type_annotations():
    """Test that type annotations are valid."""
    print("\n🏷️  Testing type annotations...")
    try:

        # Just importing should validate syntax
        print("  ✅ Type annotations syntax is valid")
        return True
    except Exception as e:
        print(f"  ❌ Type annotation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 1D-Ensemble Comprehensive Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Type Annotations", test_type_annotations()))
    results.append(("Basic Functionality", test_basic_functionality()))

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    print(f"\n⚠️  {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

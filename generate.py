from sklearn.datasets import load_iris
import pandas as pd
import os


def main(path: str = "data/iris_data.csv"):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	iris = load_iris(as_frame=True)
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	df.to_csv(path, index=False)
	print(f"Iris dataset saved to {path}")


if __name__ == "__main__":
	main()
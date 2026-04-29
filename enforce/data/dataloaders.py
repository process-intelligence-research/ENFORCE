import pandas as pd


class Dataloader:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        t_v_t_ratio: tuple[float] = (10.0, 1.0, 1.0),
    ):
        self.input_path = input_path
        self.output_path = output_path

        # Load CSV files
        try:
            self.input_data = pd.read_csv(self.input_path, header=0)
            self.output_data = pd.read_csv(self.output_path, header=0)
            print(f"Successfully loaded input from {self.input_path} and output from {self.output_path}")
            # Convert to numpy arrays
            self.input_data = self.input_data.to_numpy()
            self.output_data = self.output_data.to_numpy()
            # Split data into training, validation, and test sets
            total_samples = self.input_data.shape[0]
            train_samples = int(total_samples * t_v_t_ratio[0] / sum(t_v_t_ratio))
            val_samples = int(total_samples * t_v_t_ratio[1] / sum(t_v_t_ratio))
            test_samples = total_samples - train_samples - val_samples
            self.train_inputs = self.input_data[:train_samples]
            self.train_outputs = self.output_data[:train_samples]
            self.val_inputs = self.input_data[train_samples : train_samples + val_samples]
            self.val_outputs = self.output_data[train_samples : train_samples + val_samples]
            self.test_inputs = self.input_data[train_samples + val_samples :]
            self.test_outputs = self.output_data[train_samples + val_samples :]
            # Print the number of samples in each dataset
            print(f"Number of training samples: {train_samples}")
            print(f"Number of validation samples: {val_samples}")
            print(f"Number of test samples: {test_samples}")
            # Print the total number of samples
            print(f"Total number of samples: {total_samples}")
            # Print the ratio of training, validation, and test samples
            print(f"Training, Validation, Test ratio: {t_v_t_ratio[0]}:{t_v_t_ratio[1]}:{t_v_t_ratio[2]}")

        except Exception as e:
            print(f"Error loading CSV files: {e}")
            self.input_data = None
            self.output_data = None

    def get_data(self):
        return {
            "train_inputs": self.train_inputs,
            "train_outputs": self.train_outputs,
            "val_inputs": self.val_inputs,
            "val_outputs": self.val_outputs,
            "test_inputs": self.test_inputs,
            "test_outputs": self.test_outputs,
        }

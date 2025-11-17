import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim 
from src.model_architecture import FasterRCNNModel
from src.data_processing import GunDataset
from src.logger import get_logger
from src.custom_exception import CustomException
from torch.utils.tensorboard import SummaryWriter
import time

logger = get_logger(__name__)

model_save_path = "artifacts/models/"
os.makedirs(model_save_path, exist_ok=True)


class ModelTraining:
    
    def __init__(self, model_class, num_classes, learning_rate, epochs, dataset_path, device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        ## Tracking with TensorBoard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_directory = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_directory, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_directory)

        try:
            self.model = self.model_class(self.num_classes,self.device).model
            self.model.to(self.device)
            logger.info(f"Model initialized: {self.model_class.__name__} with {self.num_classes} classes.")

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"Optimizer initialized with learning rate: {self.learning_rate}")
        
        except Exception as e:
            logger.error(f"Error initializing model training: {str(e)}")
            raise CustomException(f"Model initialization failed: {str(e)}", sys)
        
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            dataset = GunDataset(self.dataset_path, self.device)
            dataset = torch.utils.data.Subset(dataset, range(300)) # For Local Testing Purposes

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size = 3, shuffle=True, num_workers=0, collate_fn = self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size = 3, shuffle=False, num_workers=0, collate_fn = self.collate_fn)

            logger.info(f"Dataset split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
            return train_loader, val_loader
        
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise CustomException(f"Dataset splitting failed: {str(e)}", sys)
    
    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()

            for epoch in range(self.epochs):
                logger.info(f"Starting epoch {epoch}")
                self.model.train()

                for i, (images,targets) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(images, targets)

                    if isinstance(losses, dict):
                        total_loss = 0
                        for key, value in losses.items():
                            if isinstance(value, torch.Tensor):
                                total_loss+=value
                        if total_loss==0:
                            logger.error("There was error in losses capturing")
                            raise ValueError("Total value is zero........")

                        self.writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + i)
                    
                    else:
                        total_loss=losses[0]
                    
                    total_loss.backward()
                    self.optimizer.step()
                
                self.writer.flush()

                self.model.eval()
                with torch.no_grad():
                    for images,targets in val_loader:
                        val_losses = self.model(images, targets)
                        logger.info(type(val_losses))
                        logger.info(f"VAL Losses: {val_losses}")

                    model_path = os.path.join(model_save_path, "fasterrcnn.pth")
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"Model saved successfully at {model_path}")
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise CustomException(f"Training failed: {str(e)}", sys)
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,  # Adjust based on your dataset
        learning_rate=0.001,
        epochs=1,  # Adjust based on your needs
        dataset_path="artifacts/raw/",  # Adjust to your dataset path
        device=device
    )
    training.train()
    logger.info("Model training completed successfully.")
import os
import torch
import torchaudio
import argparse
from torch.utils.data import Dataset, DataLoader
from bs_roformer import BSRoformer
import torch.optim as optim
import random
from tqdm import tqdm  # Import tqdm for progress tracking

class AudioDataset(Dataset):
    def __init__(self, root_dir, segment_duration, phase='train'):
        self.root_dir = os.path.join(root_dir, phase)
        self.segment_duration = segment_duration  # in seconds
        self.song_folders = [os.path.join(self.root_dir, folder) for folder in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, folder))]
    
    
    def __getitem__(self, idx):
        song_folder = self.song_folders[idx]
        
        # Load mixture
        mixture, sample_rate = torchaudio.load(os.path.join(song_folder, 'mixture.wav'))
        
        # Adjust segment_duration to be a multiple of 512 samples
        num_samples = int(self.segment_duration * sample_rate)
        closest_multiple_of_512 = round(num_samples / 512) * 512
        adjusted_segment_duration = closest_multiple_of_512 / sample_rate
        
        # Recalculate number of samples to crop based on adjusted_segment_duration
        num_samples = int(adjusted_segment_duration * sample_rate)
        
        # Ensure that the audio is long enough
        if mixture.shape[-1] < num_samples:
            raise ValueError(f"The audio is too short. Minimum length is {num_samples} samples.")
        
        # Load and check stems
        stems = []
        lengths = [mixture.shape[-1]]  # Store the length of mixture and all stems
        for stem in ['bass', 'drums', 'other', 'vocals']:
            stem_tensor, _ = torchaudio.load(os.path.join(song_folder, f'{stem}.wav'))
            stems.append(stem_tensor)
            lengths.append(stem_tensor.shape[-1])
            
        if len(set(lengths)) > 1:
            raise ValueError("All stems and the mixture must have the same length.")
        
        # Select a random start point for cropping
        start = random.randint(0, mixture.shape[-1] - num_samples)
        
        # If remaining samples are less than num_samples, adjust start
        if start + num_samples > mixture.shape[-1]:
            start = mixture.shape[-1] - num_samples  # adjust start to get a segment of num_samples length

        # Crop the mixture and stems
        mixture = mixture[:, start:start + num_samples]
        stems = [stem[:, start:start + num_samples] for stem in stems]
        
        # Stack stems to create the target tensor
        target = torch.stack(stems)
        return mixture, target
   
    def __len__(self):
        return len(self.song_folders)
    
def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def run_training_loop(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check CUDA availability

    # Initialize Model
    model = BSRoformer(
        dim=128,
        depth=12,
        stereo=True,
        num_stems=4,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        freqs_per_bands=(
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            12, 12, 12, 12, 12, 12, 12, 12,
            24, 24, 24, 24, 24, 24, 24, 24,
            48, 48, 48, 48, 48, 48, 48, 48,
            128, 129,
        )
    ).to(device) 

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # Setup DataLoader for training and validation
    train_dataset = AudioDataset(args.dataset, segment_duration=args.segment_duration, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = AudioDataset(args.dataset, segment_duration=args.segment_duration, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    running_sdr = {'bass': 0.0, 'drums': 0.0, 'other': 0.0, 'vocals': 0.0}

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm for progress tracking
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, targets = inputs.half().to(device), targets.half().to(device)  # Convert inputs and targets to Half and move them to the appropriate device
            optimizer.zero_grad()
            #outputs = model(inputs)

            #loss = criterion(outputs, targets)
            loss, outputs = model(inputs, target=targets)  # Compute the loss internally in the model
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"[Epoch {epoch+1}, Step {i+1}] Loss: {loss.item()}")

            # Accumulate the SDR every 10 steps instead of every step
            if i % 10 == 9:
                sdr_scores = new_sdr(targets, outputs)
                for stem_idx, stem in enumerate(['bass', 'drums', 'other', 'vocals']):
                    running_sdr[stem] += sdr_scores[:, stem_idx].mean().item()  # Accumulate the SDRs of all stems
                    
            # Log the loss and SDR every 10 steps
            if i % 10 == 9:
                avg_loss = running_loss / 10
                print(f"[Epoch {epoch+1}, Step {i+1}] Loss: {avg_loss}")
                    
                # Log the average SDR and reset for the next 10 steps
                for stem in ['bass', 'drums', 'other', 'vocals']:
                    avg_sdr = running_sdr[stem] / 10  # Since SDR is calculated every 10 steps
                    print(f"[Epoch {epoch+1}, Step {i+1}] {stem.capitalize()} SDR: {avg_sdr}")
                    running_sdr[stem] = 0.0  # Reset running SDR for the next 10 steps
                    
                running_loss = 0.0  # Reset running loss for the next 10 steps
                
        # Validation and Metric Logging
        model.eval()
        val_loss = 0.0
        val_sdr = {'bass': 0.0, 'drums': 0.0, 'other': 0.0, 'vocals': 0.0}  # Initialize SDR for validation
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.half().to(device), targets.half().to(device)  # Convert inputs and targets to Half and move them to the appropriate device
                #outputs = model(inputs)
                #loss = criterion(outputs, targets)
                loss, outputs = model(inputs, target=targets) # Compute the loss internally in the model
                val_loss += loss.item()
                
                sdr_scores = new_sdr(targets, outputs)
                for stem_idx, stem in enumerate(['bass', 'drums', 'other', 'vocals']):
                    stem_sdr = sdr_scores[:, stem_idx].mean().item()  # Assuming the SDR is calculated per sample in the batch and needs to be averaged
                    val_sdr[stem] += stem_sdr  # Accumulate the SDRs of all stems

        # Log the validation metrics
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss / len(test_loader)}")
        for stem in ['bass', 'drums', 'other', 'vocals']:
            avg_val_sdr = val_sdr[stem] / len(test_loader)
            print(f"[Epoch {epoch+1}] Validation {stem.capitalize()} SDR: {avg_val_sdr}")

        
        if (epoch + 1) % args.save_every == 0:
            # Check if save directory exists; if not, create it
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='Train BSRoformer Model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every n epochs')
    parser.add_argument('--segment_duration', type=float, default=3.0, help='Segment duration in seconds')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save the model weights')
    args = parser.parse_args()

    # Run training loop
    run_training_loop(args)
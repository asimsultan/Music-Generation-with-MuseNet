
import torch
import argparse
from openai import MuseNet
from datasets import load_dataset
from utils import get_device, preprocess_midi, MusicDataset
from midiutil import MIDIFile

def main(model_path, data_path, output_path):
    # Load Model
    model = MuseNet.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = load_dataset('midi', data_files={'validation': data_path})
    preprocessed_datasets = dataset.map(preprocess_midi, batched=True)

    # DataLoader
    eval_dataset = MusicDataset(preprocessed_datasets['validation'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device, output_path):
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += labels.size(0)

                # Generate Music
                generated_sequence = model.generate(input_ids=input_ids, max_length=512)

                # Convert to MIDI
                midi = MIDIFile(1)
                track = 0
                time = 0
                midi.addTrackName(track, time, "Generated Track")
                midi.addTempo(track, time, 120)

                for i, note in enumerate(generated_sequence[0]):
                    midi.addNote(track, 0, note.item(), time + i, 1, 100)

                with open(f"{output_path}/generated_{i}.mid", "wb") as output_file:
                    midi.writeFile(output_file)

        avg_loss = total_loss / total_samples
        return avg_loss

    # Evaluate
    avg_loss = evaluate(model, eval_loader, device, output_path)
    print(f'Average Loss: {avg_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the MIDI files')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated MIDI files')
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_path)

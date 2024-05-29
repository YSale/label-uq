import torch
import time
import tqdm
from sklearn.metrics import confusion_matrix


def train_supervised_neural_network(model, model_name, results_path, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test']}

    # Create a temporary directory to save training checkpoints
    best_model_params_path = f'{results_path}models/{model_name}.pt'

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            y_true = []
            y_pred = []

            # Iterate over data.
            with tqdm.tqdm(dataloaders[phase], unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                for inputs, labels in tepoch:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'test':
                            lbl = labels.data.cpu().numpy()
                            prd = preds.data.cpu().numpy()
                            y_true.extend(lbl)
                            y_pred.extend(prd)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    corrects = torch.sum(preds == labels.data).cpu()
                    running_corrects += corrects
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * (corrects.numpy()/inputs.size(0)))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)

            if phase == 'test':
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    print('Confusion Matrix: ')
                    conf_mat = confusion_matrix(y_true, y_pred)
                    print(conf_mat)

                if epoch % 10 == 0:
                    print('Confusion Matrix: ')
                    conf_mat = confusion_matrix(y_true, y_pred)
                    print(conf_mat)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model

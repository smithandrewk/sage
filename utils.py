import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import copy
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def count_params(model):
    return sum([p.flatten().size()[0] for p in list(model.parameters())])

def evaluate(dataloader,model,criterion,device='cuda'):
    model.eval()
    model.to(device)
    criterion.to(device)
    from tqdm import tqdm
    with torch.no_grad():
        loss_total = 0
        y_true = []
        y_pred = []
        for Xi,yi in dataloader:
            Xi,yi = Xi.to(device),yi.to(device)
            logits = model(Xi)
            loss = criterion(logits,yi)
            loss_total += loss.item()

            y_true.append(yi.argmax(axis=1).cpu())
            y_pred.append(logits.softmax(dim=1).argmax(axis=1).cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    return loss_total/len(dataloader),y_true,y_pred
def predict(model,dataloader):
    ypreds = []
    for Xi,_ in dataloader:
        ypreds.append(model(Xi).softmax(axis=1).argmax(axis=1))
    return torch.cat(ypreds)
def train(state,trainloader,devloader,**kwargs):
    last_time = time()
    state['model'].to(state['device'])
    state['criterion'].to(state['device'])
    pbar = tqdm(range(state['epochs']))
    epochs_without_improvement = 0

    for _ in pbar:
        state['model'].train()

        all_trues = []
        all_preds = []
        loss_total = 0
        for Xi,yi in trainloader:
            Xi,yi = Xi.to(state['device']),yi.to(state['device'])
            logits = state['model'](Xi)
            loss = state['criterion'](logits,yi)
            state['optimizer'].zero_grad()
            loss.backward()
            state['optimizer'].step()
            loss_total += loss.item()
        
            # Collect predictions and labels for F1 score calculation
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_trues.append(yi.argmax(dim=1).detach().cpu().numpy())
        
        state['trainlossi'].append(loss_total/len(trainloader))

        # Calculate training F1 score using collected predictions and labels
        all_trues = np.concatenate(all_trues)
        all_preds = np.concatenate(all_preds)
        state['trainf1i'].append(f1_score(all_trues, all_preds, average='macro'))

        state['model'].eval()
        with torch.no_grad():
            loss,y_true,y_pred = evaluate(dataloader=devloader,model=state['model'],criterion=state['criterion'],device=state['device'])
            state['devlossi'].append(loss)
            state['devf1i'].append(f1_score(y_true,y_pred,average='macro'))

        state['scheduler'].step(state['devlossi'][-1])

        # TODO : could be potentiially early stopper.step()
        # Track best dev loss and save model weights and early stopping
        if state['devlossi'][-1] < state['best_dev_loss']:
            state['best_dev_loss'] = state['devlossi'][-1]
            state['best_dev_loss_epoch'] = len(state['devlossi'])-1
            state['best_model_wts_dev_loss'] = copy.deepcopy(state['model'].state_dict())
            epochs_without_improvement = 0
        elif epochs_without_improvement >= state['patience']:
            state['early_stopping'] = True
            break
        else:
            epochs_without_improvement += 1

        # Track best dev f1 and save model weights
        if state['devf1i'][-1] > state['best_dev_f1']:
            state['best_dev_f1'] = state['devf1i'][-1]
            state['best_dev_f1_epoch'] = len(state['devf1i'])-1
            state['best_model_wts_dev_f1'] = copy.deepcopy(state['model'].state_dict())
        
        # Moving average for execution time of an epoch
        state['execution_time'] = (state['execution_time'] + (time() - last_time))/2
        pbar.set_description(f'train: {state["trainlossi"][-1]:.4f}, dev: {state["devlossi"][-1]:.4f}, best_dev: {state["best_dev_loss"]:.4f}')
        yield state
        
    return state

def plot_loss(state,experiment_path):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4.2))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    axes[0].plot(state['trainlossi'], color=colors[0], linestyle='-', label='train')
    axes[0].plot(state['devlossi'], color=colors[0], linestyle=':', label='val')
    axes[0].axvline(state['best_dev_loss_epoch'], color=colors[0], linestyle=':')
    axes[0].axhline(state['best_dev_loss'], color=colors[0], linestyle=':')

    axes[1].plot(state['trainf1i'], color=colors[0], linestyle='-', label='train')
    axes[1].plot(state['devf1i'], color=colors[0], linestyle=':', label='val')
    axes[1].axvline(state['best_dev_loss_epoch'], color=colors[0], linestyle=':')
    axes[1].axvline(state['best_dev_f1_epoch'], color=colors[1], linestyle=':')
    axes[1].axhline(state['devf1i'][state['best_dev_loss_epoch']], color=colors[0], linestyle=':')
    axes[1].axhline(state['devf1i'][state['best_dev_f1_epoch']], color=colors[1], linestyle=':')

    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{experiment_path}/loss.jpg')
    plt.savefig(f'loss.jpg')
    plt.close()

def plot_train_and_test_loss_in_plotly(df):
    # Extract trainloss and testloss
    trainlossi = []
    testlossi = []
    for trainloss, testloss in zip(df['trainlossi'].iloc, df['testlossi'].iloc):
        trainlossi.append(trainloss)
        testlossi.append(testloss)

    # Convert to DataFrame
    trainloss_df = pd.DataFrame(trainlossi).T
    testloss_df = pd.DataFrame(testlossi).T

    # Create Plotly figure
    fig = go.Figure()

    # Define color palette
    colors = px.colors.qualitative.Plotly

    # Add train and test loss traces
    for i, col in enumerate(trainloss_df.columns):
        color = colors[i % len(colors)]  # Cycle through colors if more columns than colors
        fig.add_trace(go.Scatter(
            x=trainloss_df.index,
            y=trainloss_df[col],
            mode='lines',
            name=f'{df.iloc[i,5:7]}',
            line=dict(dash='solid', color=color)
        ))
        fig.add_trace(go.Scatter(
            x=testloss_df.index,
            y=testloss_df[col],
            mode='lines',
            line=dict(dash='dash', color=color)
        ))

    # Update layout
    fig.update_layout(
        title='Train and Test Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='Loss Type'
    )

    # Show plot in browser
    fig.show(renderer='browser')

def plot_loss_curves(plot_df,moving_window_length,lstm=False):
    plot_df.loc[plot_df['best_dev_f1'] == '','best_dev_f1'] = 0
    plot_df['best_dev_loss'] = plot_df['best_dev_loss'].astype(float)
    plot_df['best_dev_f1'] = plot_df['best_dev_f1'].astype(float)
    plot_df['best_dev_loss'] = plot_df['best_dev_loss'].apply(lambda x: f'{x:.4f}')
    plot_df['best_dev_f1'] = plot_df['best_dev_f1'].apply(lambda x: f'{x:.4f}' if not pd.isnull(x) else '')

    # Function to compute moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Extract trainloss and testloss
    trainlossi = []
    testlossi = []
    for trainloss, testloss in zip(plot_df['trainlossi'].iloc, plot_df['testlossi'].iloc):
        trainlossi.append(trainloss[moving_window_length:])
        testlossi.append(moving_average(testloss[moving_window_length:], window_size=moving_window_length))  # Applying moving average

    # Convert to DataFrame
    trainloss_df = pd.DataFrame(trainlossi).T
    testloss_df = pd.DataFrame(testlossi).T

    # Use seaborn for better aesthetics
    sns.set(style="whitegrid")

    # Create a figure with a GridSpec layout
    fig = plt.figure(figsize=(9, 16))
    gs = GridSpec(2,1, width_ratios=[1],height_ratios=[2,1])

    # Plotting the loss curves
    ax0 = plt.subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(trainloss_df.columns)))

    handles = []
    labels = []

    for i, color in enumerate(colors):
        train_line, = ax0.plot(trainloss_df[i], label=f'Experiment {i+1} - Training Loss', color=color, linestyle='-')
        val_line, = ax0.plot(testloss_df[i], label=f'Experiment {i+1} - Validation Loss (Moving Avg)', color=color, linestyle='--')
        
        handles.append(train_line)
        labels.append(f'Experiment {i+1}')

    ax0.set_title('Training and Validation Loss Curves', fontsize=20, fontweight='bold', pad=20)
    ax0.set_xlabel('Epoch', fontsize=16, labelpad=15)
    ax0.set_ylabel('Loss', fontsize=16, labelpad=15)
    ax0.set_yscale('log')
    # ax0.set_ylim([0.00001,.4])
    
    ax0.tick_params(axis='both', which='major', labelsize=14)
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax0.legend(handles=handles, labels=labels, fontsize=12, loc='upper right', frameon=True, framealpha=1, shadow=True, borderpad=1)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    # Plotting the hyperparameters
    ax1 = plt.subplot(gs[1])
    if lstm:
        hyperparameters = plot_df[['best_dev_loss', 'sequence_length','hidden_size','num_layers','dropout','frozen_encoder','robust']]
    else:
        hyperparameters = plot_df[['best_dev_loss','batch_size','dropout']]

    # Add a color column to the hyperparameters DataFrame
    color_column = [''] * len(hyperparameters)
    for i, color in enumerate(colors):
        color_column[i] = ''  # Leave this as an empty string, we'll color the cells separately

    hyperparameters.insert(0, 'Color', color_column)

    cell_text = hyperparameters.values
    columns = hyperparameters.columns
    table = ax1.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center', colWidths=[0.1]*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Adjusted font size for better fit
    table.scale(2,2)  # Adjusted scale for better fit


    # Set color for the color column cells
    for (i, j), cell in table.get_celld().items():
        if j == 0 and i > 0:  # Skip header
            cell.set_facecolor(colors[i-1])
            cell.set_text_props(text='')  # Ensure the text is empty

    ax1.axis('off')
    ax1.set_title('Hyperparameter Settings', fontsize=20, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('losses.jpg',dpi=200,bbox_inches='tight')
    plt.close()
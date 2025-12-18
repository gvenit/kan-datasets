import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(
        cm, 
        class_names, 
        title='Confusion Matrix', 
        normalize=False, 
        figsize=(6, 4), 
        cmap='Blues', 
        save_path=None
    ):
    """
    Plot confusion matrix with customizable options.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        title: Plot title
        normalize: If True, normalize the confusion matrix
        figsize: Figure size tuple
        cmap: Colormap for the heatmap
        save_path: Optional path to save the figure
        
    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    if normalize:
        # Normalize confusion matrix, handling zero-sum rows
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Replace NaN values (from rows with zero predictions) with 0
            cm = np.nan_to_num(cm, nan=0.0)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax, square=True, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

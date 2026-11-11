{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMl+WLnp7pRNrQ0lEBnHdSd",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ashish-1911/Machine-Learning/blob/main/bigram_language_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "53wg15uvzaO4",
        "outputId": "8ababb70-656b-419e-aeba-07323686ffb9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Step 0: Loss = 1.3863\n",
            "Step 50: Loss = 0.1121\n",
            "Step 100: Loss = 0.0518\n",
            "Step 300: Loss = 0.0159\n",
            "Step 600: Loss = 0.0078\n",
            "Generated: i like ai\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import numpy as np\n",
        "\n",
        "# 1. Vocabulary and training data\n",
        "vocab = [\"i\", \"like\", \"ai\", \"<eos>\"]\n",
        "to_id = {word: i for i, word in enumerate(vocab)}\n",
        "\n",
        "sequence = [\"i\", \"like\", \"ai\", \"<eos>\"]\n",
        "ids = np.array([to_id[word] for word in sequence])\n",
        "\n",
        "# Input token -> next target token\n",
        "x = ids[:-1]\n",
        "y = ids[1:]\n",
        "\n",
        "V = len(vocab)\n",
        "W = np.zeros((V, V))\n",
        "\n",
        "learning_rate = 0.5\n",
        "\n",
        "# 2. Train the model\n",
        "for step in range(601):\n",
        "\n",
        "    # Forward pass: output logits\n",
        "    logits = W[x]\n",
        "\n",
        "    # Numerically stable softmax\n",
        "    logits = logits - logits.max(axis=1, keepdims=True)\n",
        "    exp_logits = np.exp(logits)\n",
        "    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)\n",
        "\n",
        "    # Cross-entropy loss\n",
        "    loss = -np.log(\n",
        "        probs[np.arange(len(y)), y]\n",
        "    ).mean()\n",
        "\n",
        "    if step in [0, 50, 100, 300, 600]:\n",
        "        print(f\"Step {step}: Loss = {loss:.4f}\")\n",
        "\n",
        "    # 3. Backward pass\n",
        "    grad_logits = probs.copy()\n",
        "    grad_logits[np.arange(len(y)), y] -= 1\n",
        "    grad_logits /= len(y)\n",
        "\n",
        "    grad_W = np.zeros_like(W)\n",
        "    np.add.at(grad_W, x, grad_logits)\n",
        "\n",
        "    # 4. Gradient descent update\n",
        "    W -= learning_rate * grad_W\n",
        "\n",
        "# 5. Generate text\n",
        "current = to_id[\"i\"]\n",
        "generated = [\"i\"]\n",
        "\n",
        "for _ in range(10):\n",
        "    next_id = int(np.argmax(W[current]))\n",
        "    next_word = vocab[next_id]\n",
        "\n",
        "    if next_word == \"<eos>\":\n",
        "        break\n",
        "\n",
        "    generated.append(next_word)\n",
        "    current = next_id\n",
        "\n",
        "print(\"Generated:\", \" \".join(generated))\n"
      ]
    }
  ]
}
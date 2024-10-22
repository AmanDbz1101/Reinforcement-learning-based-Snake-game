# Reinforcement-Learning-Based-Snake-Game
This model includes training the snake in Snake game and also using the trained model as a bot.

## 1. Clone the repository
Use the following command to clone this repository to your local machine:

```bash
git clone https://github.com/AmanDbz1101/Reinforcement-learning-based-Snake-game.git
```
## 2. Navigate to the Project Direcotry

```bash
cd Reinforcement-learning-based-Snake-game
```
## 3. Installation

To get started, follow the instructions below to install the necessary dependencies and set up the project.
```bash
pip install -r requirements.txt
```
## 4. Train Model
To train the model, run the agent.py file in Training folder.
```bash
cd Training
```
```bash
python -u agent.py
```
This will save the .weights.h5 file in model folder.
## 5. Use trained model as bot 
To use the trained model, use model.load_weights to load the data from saved file. In this project, an example is given. To use that run agent.py file in bot_game folder.
```bash
python -u agent.py
```
# Note: Remember to change the directory before running this code.
To get out from Training folder
```bash
cd ..
```
To get into bot_game folder
```bash
cd bot_game
```


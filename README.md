# Reinforcement-Learning-Based-Snake-Game
This project includes training model for the snake in Snake game and also using the trained model as a bot.

### Here is an example of before training much and after training for some time:
<p align="center">
  <img src="https://github.com/user-attachments/assets/b1b455f4-b68f-437b-a9f9-fc019a18fa9b" width="45%" />
  <img src="https://github.com/user-attachments/assets/560174e4-6815-4d16-a0d8-797b03ad31a2" width="45%" />
</p>



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
### Note: Remember to change the directory before running this code.
To get out from Training folder
```bash
cd ..
```
To get into bot_game folder
```bash
cd bot_game
```


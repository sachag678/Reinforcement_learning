import streamlit as st
import plotly.express as px
import numpy as np

from Agents.GreedyStateActionValueAgent import GreedyStateActionValueAgent
from Agents.GreedyStateValueAgent import GreedyStateValueAgent
from Run import train, moving_average, test
from Tabular_methods.Action_State_Value.QlearningTabular import QLearningTabular
from Tabular_methods.Action_State_Value.SARSATabular import SARSATabular
from Tabular_methods.State_Value.MonteCarloTabular import MonteCarloTabular
from Tabular_methods.State_Value.TemporalDifferenceTabular import TemporalDifferenceTabular

st.set_page_config(layout="wide")
st.header("Tabular RL Experiments")

with st.sidebar:
    learning_algorithm = st.selectbox(label="Select a type of learning algorithm: ", index=2,
                                      options=["SARSA", "QLearning", "TDLearning", "MonteCarlo"])
    num_episodes = int(st.number_input('Select number of training episodes: ', value=100, min_value=1, step=10))
    epsilon = st.number_input('Select epsilon for greedy epsilon search: ', value=0.9, min_value=0.0, max_value=1.0,
                              step=0.1)
    gamma = st.number_input('Select gamma (discounting factor for future reward): ', value=0.95, min_value=0.0,
                            max_value=1.0, step=0.1)
    alpha = st.number_input("Select learning rate (alpha): ", value=0.01, max_value=1.0, min_value=0.0001, step=1e-4,
                            format="%.4f")

if learning_algorithm == "TDLearning":
    model = TemporalDifferenceTabular(alpha=alpha, gamma=gamma)
    agent = GreedyStateValueAgent(model=model, epsilon=epsilon)
elif learning_algorithm == "MonteCarlo":
    model = MonteCarloTabular(alpha=alpha, gamma=gamma)
    agent = GreedyStateValueAgent(model=model, epsilon=epsilon)
elif learning_algorithm == "QLearning":
    model = QLearningTabular(alpha=alpha, gamma=gamma)
    agent = GreedyStateActionValueAgent(model=model, epsilon=epsilon)
else:
    model = SARSATabular(alpha=alpha, gamma=gamma)
    agent = GreedyStateActionValueAgent(model=model, epsilon=epsilon)

agent, episode_rewards = train(num_episodes, agent)
agent.epsilon = 0
output_str = test(agent, verbose=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Moving Average of Episode Rewards")
    y = moving_average(episode_rewards, 10)
    fig = px.line(y=y, x=[i for i in range(1, len(y) + 1)])
    fig.update_layout(xaxis_title='Timestep',
                      yaxis_title='Reward')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if learning_algorithm in ["TDLearning", "MonteCarlo"]:
        st.subheader("State Value Plot")
        fig = px.imshow(agent.model.get_model_fig_data(), text_auto=True)
        fig.layout.coloraxis.showscale = False
        fig.update_yaxes(nticks=6)
        st.plotly_chart(fig)
    else:
        st.subheader("State-Action Plot")
        data = agent.model.get_model_fig_data()
        fig = px.imshow(np.ones((4, 4)))
        fig.update_yaxes(nticks=6)
        fig.layout.coloraxis.showscale = False
        for i in range(4):
            for j in range(4):
                if i == 1 and j == 2:
                    action_str = "Dragon"
                elif i == 0 and j == 3:
                    action_str = "Gold"
                else:
                    action_str = data[i][j]
                fig.add_annotation(x=j, y=i, text=action_str, showarrow=False)
        st.plotly_chart(fig)

st.subheader("Testing learned model!")
st.write("Player is at (3, 0), Gold is at (0, 3), Dragon is at (1, 2)")
st.write(output_str)

# IPL 2025 Monte Carlo Outcome Simulator

## 1. Overview

This project uses Monte Carlo simulations to predict the potential outcomes of the IPL 2025 season based on the schedule, completed match results, and defined probabilistic models for remaining matches and playoffs.

It simulates the remainder of the season thousands (or millions) of times to estimate:
*   The probability of each team qualifying for the Top 4 (playoffs).
*   The probability of each team winning the championship.
*   The likelihood of teams reaching specific playoff stages (Qualifier 1, Eliminator, Qualifier 2, Final).
*   The distribution of final points for each team (Average, Median, Mode).

## 2. Prerequisites

*   **Python 3.x**
*   **Python Libraries:**
    *   `pandas`
    *   `numpy`
    *   `matplotlib` (Optional, used for generating plots if uncommented)

## 3. Installation

1.  **Get the Files:** Ensure you have the following two files in the same directory:
    *   `ipl_2025_mc.py` (The Python script)
    *   `ipl-2025-UTC.csv` (The schedule and results data file)
2.  **Install Libraries:** Open your terminal or command prompt and install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib
    ```

## 4. How to Run & Update Results

### Running the Simulation

1.  Navigate to the directory containing the script and CSV file in your terminal.
2.  Run the script using Python:
    ```bash
    python ipl_2025_mc.py
    ```
3.  The script will first display the **Current Standings** based on the results already entered in the script.
4.  It will then run the specified number of Monte Carlo simulations (default is 1,000,000 in the provided code, adjustable via `num_simulations` variable).
5.  Finally, it will print the calculated probabilities and statistics to the console.

### Updating Match Results

As the IPL season progresses, you need to update the script with the actual results of completed matches for the simulations to be accurate.

1.  **Open the Script:** Edit the `ipl_2025_mc.py` file in a text editor or IDE.
2.  **Locate the Update Section:** Find the section marked with:
    ```python
    # ==============================================================================
    # == UPDATE COMPLETED MATCH RESULTS HERE ==
    # ==============================================================================
    ```
3.  **Add/Update Results:** For each completed match, use the `update_result` function. You need the `match_number` and the `winner` (using the team abbreviations defined in the `team_abbreviations` dictionary).
    *   **For a clear winner:**
        ```python
        # Example: Match 47 won by Rajasthan Royals (RR)
        df_abbreviated = update_result(df_abbreviated, match_number=47, winner='RR')
        ```
    *   **For a No Result (e.g., washout):**
        ```python
        # Example: Match 48 was a No Result
        df_abbreviated = update_result(df_abbreviated, match_number=48, winner='NR')
        # or alternatively:
        # df_abbreviated = update_result(df_abbreviated, match_number=48, winner=None)
        ```
    *   **Ensure you use the correct abbreviations:** KKR, RCB, SRH, RR, CSK, MI, DC, LSG, GT, PBKS.
4.  **Save the Script:** Save the changes to `ipl_2025_mc.py`.
5.  **Re-run:** Execute the script again (`python ipl_2025_mc.py`) to get updated predictions based on the new results.

## 5. Monte Carlo Simulation & Code Logic

### What are Monte Carlo Simulations?

Monte Carlo simulation is a computational technique that uses repeated random sampling to obtain numerical results. It's particularly useful for modeling phenomena with significant uncertainty. In this context, the uncertainty lies in the outcome of future cricket matches.

Instead of predicting a single outcome, we simulate the *entire* remaining season many times. Each simulation represents one possible future. By analyzing the distribution of outcomes across thousands of these simulated futures, we can estimate the probability of different events (like a team winning).

### How the Code Works

1.  **Load Data:** Reads the schedule and basic team info from `ipl-2025-UTC.csv`.
2.  **Apply Known Results:** Updates the schedule DataFrame based on the results hardcoded in the `UPDATE COMPLETED MATCH RESULTS HERE` section.
3.  **Calculate Initial State:** Determines the current points table based *only* on the completed matches entered.
4.  **Simulation Loop:** Runs `num_simulations` times:
    *   **Copy Initial State:** Starts each simulation run with the current, real-world points table.
    *   **Simulate Remaining League Matches:** For each match yet to be played in the league stage, it randomly assigns a winner with a **50/50 probability** (see Biases section). Points are updated accordingly for that simulation run.
    *   **Determine League Standings:** After simulating all league matches, teams are ranked based on Points, then Wins (Note: NRR is ignored for simplicity).
    *   **Simulate Playoffs:** The Top 4 teams enter a simulated playoff based on the standard IPL format (Qualifier 1, Eliminator, Qualifier 2, Final). Playoff match outcomes are determined using **biased probabilities** (see Biases section).
    *   **Record Outcomes:** For each simulation, the script records which teams finished in the Top 4, which teams participated in each playoff stage, the final points tally for each team, and the ultimate tournament winner.
5.  **Aggregate Results:** After all simulations are complete, the script counts how many times each outcome occurred.
6.  **Calculate Probabilities & Stats:** Divides the counts by `num_simulations` to get probabilities (e.g., Top 4 %, Championship %) and calculates descriptive statistics for points distributions (Mean, Median, Mode).
7.  **Display Results:** Prints the findings to the console.

## 6. Biases Introduced (and Why)

The simulation incorporates specific biases to make the predictions more realistic than pure randomness:

1.  **Regular Season Matches (50/50 Probability):**
    *   **Bias:** Assumes any team has an equal chance of winning any remaining league match.
    *   **Reasoning:** This is a simplification reflecting the high unpredictability of T20 cricket. It doesn't account for team strength, form, or venue during the simulation of these matches, but team strength *is* implicitly factored in when determining the initial playoff seeding based on points accumulated.
2.  **Playoff Matches (Points-Based Bias):**
    *   **Bias:** In Qualifier 1, Eliminator, and Qualifier 2, the team entering the match with more points from the league stage is given a slightly higher probability of winning. The `bias_factor` (e.g., 0.1 or 0.2) controls the strength of this advantage. Probabilities are capped (e.g., between 20% and 80%) to prevent extreme skew and allow for upsets.
    *   **Reasoning:** To reflect that teams finishing higher in the table generally have demonstrated better performance and might have a slight edge in knockout games, while still acknowledging the possibility of upsets.
3.  **Final Match (Historical Bias):**
    *   **Bias:** The simulation gives the winner of Qualifier 1 a significantly higher probability of winning the Final, based on historical IPL data (approx. 10 wins in 14 finals for the Q1 winner). A small adjustment based on the finalists' league points is also included.
    *   **Reasoning:** To incorporate the strong historical trend observed in the IPL playoff format where the team taking the direct route from Qualifier 1 often has an advantage (rest, momentum, potentially playing at a preferred venue determined earlier).

## 7. How to Interpret the Results

The script outputs several tables:

*   **Current Standings:** Shows the points table based *only* on the results you've entered into the script. This is the starting point for all simulations.
*   **Estimated Top 4 Qualification Probabilities:**
    *   `Qualification %`: The percentage of simulations where that team finished in the Top 4 of the league standings. A higher percentage indicates a stronger likelihood of making the playoffs.
*   **CHAMPIONSHIP PROBABILITIES:**
    *   `Championship %`: The percentage of simulations where that team won the Final, incorporating all playoff and historical biases. This is the model's prediction for the ultimate winner.
*   **PLAYOFF STAGE ADVANCEMENT PROBABILITIES:**
    *   `Qualifier 1 %`: Chance of finishing 1st or 2nd in the league.
    *   `Eliminator %`: Chance of finishing 3rd or 4th in the league.
    *   `Qualifier 2 %`: Chance of playing in the Qualifier 2 match (either as Q1 Loser or Eliminator Winner).
    *   `Final %`: Chance of reaching the Grand Final.
    *   `Champion %`: Same as the Championship Probability table, shown here for comparison across stages. This table helps visualize the most likely paths through the playoffs for each team.
*   **Estimated Final Points Distribution Statistics:**
    *   `Avg Pts`: The average number of points a team finished with across all simulations.
    *   `Median Pts`: The midpoint of the points distribution (50% of simulations finished with more points, 50% with fewer). Less sensitive to extreme outlier results than the average.
    *   `Mode Pts`: The most frequently occurring final points tally for that team across all simulations. Indicates the most common single points outcome.

Result of Sim as on 28th April 2025-

                                       ![image](https://github.com/user-attachments/assets/fa97dade-1761-4f38-89ed-0e37600c341d)


**Important Note:** These are probabilistic estimates based on the model's assumptions and the results entered. They are not guarantees and do not account for factors like player injuries, sudden changes in form, specific pitch conditions, or precise Net Run Rate calculations.

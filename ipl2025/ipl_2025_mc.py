import numpy as np
import pandas as pd
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Load the dataframe
try:
    df_orig = pd.read_csv('ipl-2025-UTC.csv')
    # Filter out playoff matches (71-74) which have "To be announced" teams
    df_orig = df_orig[df_orig['Match Number'] <= 70]
    df_orig = df_orig.drop(columns=['Round Number', 'Date', 'Location'])

    team_abbreviations = {
        'Kolkata Knight Riders': 'KKR', 'Royal Challengers Bengaluru': 'RCB',
        'Sunrisers Hyderabad': 'SRH', 'Rajasthan Royals': 'RR',
        'Chennai Super Kings': 'CSK', 'Mumbai Indians': 'MI',
        'Delhi Capitals': 'DC', 'Lucknow Super Giants': 'LSG',
        'Gujarat Titans': 'GT', 'Punjab Kings': 'PBKS'
    }
    df_abbreviated = df_orig.copy()
    df_abbreviated['Home Team'] = df_abbreviated['Home Team'].map(team_abbreviations).fillna(df_abbreviated['Home Team'])
    df_abbreviated['Away Team'] = df_abbreviated['Away Team'].map(team_abbreviations).fillna(df_abbreviated['Away Team'])

    if 'Result' not in df_abbreviated.columns:
        df_abbreviated['Result'] = ''
    df_abbreviated['Result'] = df_abbreviated['Result'].fillna('')

except FileNotFoundError:
    print("Error: Schedule CSV file not found. Please check the path.")
    # Create a dummy dataframe for demonstration if file not found
    data = {'Match Number': range(1, 71),
            'Home Team': ['CSK', 'DC', 'KKR', 'PBKS', 'SRH', 'RCB', 'LSG', 'GT', 'RR', 'MI'] * 7,
            'Away Team': ['RCB', 'CSK', 'SRH', 'RR', 'KKR', 'PBKS', 'GT', 'MI', 'DC', 'LSG'] * 7,
            'Result': [''] * 70} # Empty results initially
    df_abbreviated = pd.DataFrame(data)
    print("Using dummy data for demonstration.")

# Get the actual team list from the dataframe (excluding "To be announced")
teams = sorted(list(set(df_abbreviated['Home Team']).union(set(df_abbreviated['Away Team']))))
# Make sure "To be announced" is not in the teams list
if 'To be announced' in teams:
    teams.remove('To be announced')

# Define a function to update the result for a given match number
def update_result(df, match_number, winner):
    # Ensure winner abbreviation is used if full name provided
    winner_abbr = team_abbreviations.get(winner, winner)
    if match_number in df['Match Number'].values:
        df.loc[df['Match Number'] == match_number, 'Result'] = winner_abbr
    else:
        print(f"Warning: Match Number {match_number} not found in schedule.")
    return df

# ==============================================================================
# == UPDATE COMPLETED MATCH RESULTS HERE ==
# ==============================================================================
# Match results updated through match 42

df_abbreviated = update_result(df_abbreviated, match_number=1, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=2, winner='SRH')
df_abbreviated = update_result(df_abbreviated, match_number=3, winner='CSK')
df_abbreviated = update_result(df_abbreviated, match_number=4, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=5, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=6, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=7, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=8, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=9, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=10, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=11, winner='RR')
df_abbreviated = update_result(df_abbreviated, match_number=12, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=13, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=14, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=15, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=16, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=17, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=18, winner='RR')
df_abbreviated = update_result(df_abbreviated, match_number=19, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=20, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=21, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=22, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=23, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=24, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=25, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=26, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=27, winner='SRH')
df_abbreviated = update_result(df_abbreviated, match_number=28, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=29, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=30, winner='CSK')
df_abbreviated = update_result(df_abbreviated, match_number=31, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=32, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=33, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=34, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=35, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=36, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=37, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=38, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=39, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=40, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=41, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=42, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=43, winner='SRH')


# ==============================================================================

# --- Calculate Initial Points Table based on CURRENT completed matches ---
completed_matches = df_abbreviated[df_abbreviated['Result'] != ''].copy()

# Initialize a points dictionary (base state)
initial_points_table = {team: {'Matches': 0, 'Wins': 0, 'Losses': 0, 'Points': 0} for team in teams}

# Calculate wins, losses, and points from completed matches
for _, row in completed_matches.iterrows():
    home = row['Home Team']
    away = row['Away Team']
    winner = row['Result']

    if home in initial_points_table: initial_points_table[home]['Matches'] += 1
    if away in initial_points_table: initial_points_table[away]['Matches'] += 1

    if winner == home:
        if home in initial_points_table:
            initial_points_table[home]['Wins'] += 1
            initial_points_table[home]['Points'] += 2
        if away in initial_points_table: initial_points_table[away]['Losses'] += 1
    elif winner == away:
        if away in initial_points_table:
            initial_points_table[away]['Wins'] += 1
            initial_points_table[away]['Points'] += 2
        if home in initial_points_table: initial_points_table[home]['Losses'] += 1

# --- Display Current Standings ---
print("=" * 60)
print(f"Current Standings ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})")
print(f"Based on {len(completed_matches)} completed matches:")
initial_points_df = pd.DataFrame.from_dict(initial_points_table, orient='index')
initial_points_df = initial_points_df.reset_index().rename(columns={'index': 'Team'})
print(initial_points_df.sort_values(by=['Points', 'Wins'], ascending=[False, False]).to_string(index=False))
print("=" * 60)

# --- Monte Carlo Simulation Part ---

# Get remaining matches
remaining_matches = df_abbreviated[df_abbreviated['Result'] == ''].copy()
num_remaining_matches = len(remaining_matches)

if num_remaining_matches == 0:
    print("No remaining matches to simulate. Final standings are above.")
else:
    # --- Monte Carlo Settings ---
    num_simulations = 1000000
    print(f"Running Monte Carlo simulation with {num_simulations:,} scenarios...")
    print(f"Simulating {num_remaining_matches} regular season matches + playoffs")
    start_time = time.time()

    # Initialize counters for various metrics
    top_4_qualifications = {team: 0 for team in teams}
    total_points_distribution = {team: [] for team in teams}

    # New counters for playoff tracking
    qualifier1_appearances = {team: 0 for team in teams}  # 1st and 2nd place
    eliminator_appearances = {team: 0 for team in teams}  # 3rd and 4th place
    qualifier2_appearances = {team: 0 for team in teams}  # Loser of Q1 + Winner of Eliminator
    final_appearances = {team: 0 for team in teams}       # Winner of Q1 + Winner of Q2
    tournament_winners = {team: 0 for team in teams}      # Winner of Final

    # --- Helper function for points-based playoff match simulation ---
    def simulate_playoff_match(team1, team2, team1_points, team2_points, bias_factor=1.0):
        """
        Simulate a playoff match with weighted probabilities based on points.

        Args:
            team1, team2: The two teams playing
            team1_points, team2_points: Points earned in league stage
            bias_factor: Controls how much points affect probability (higher = stronger bias)

        Returns:
            Winner of the match
        """
        total_points = team1_points + team2_points
        if total_points == 0:
            team1_prob = 0.5
        else:
            raw_ratio = team1_points / total_points
            team1_prob = 0.5 + (raw_ratio - 0.5) * bias_factor
            team1_prob = max(0.2, min(0.8, team1_prob))
        return team1 if random.random() < team1_prob else team2

    # --- Main Simulation Loop ---
    for i in range(num_simulations):
        # Start with a deep copy of the initial points table for this run
        current_scenario_points = copy.deepcopy(initial_points_table)

        # Simulate each remaining match with a random outcome
        for _, match_info in remaining_matches.iterrows():
            home = match_info['Home Team']
            away = match_info['Away Team']

            # Skip matches where teams are "To be announced"
            if home == "To be announced" or away == "To be announced":
                continue

            # Randomly determine winner (50/50 chance)
            winner = random.choice([home, away])

            # Update points, wins, losses, matches for this run
            current_scenario_points[home]['Matches'] += 1
            current_scenario_points[away]['Matches'] += 1

            if winner == home:
                current_scenario_points[home]['Wins'] += 1
                current_scenario_points[home]['Points'] += 2
                current_scenario_points[away]['Losses'] += 1
            else:
                current_scenario_points[away]['Wins'] += 1
                current_scenario_points[away]['Points'] += 2
                current_scenario_points[home]['Losses'] += 1

        # Scenario complete for this run, determine final standings
        final_standings = []
        for team, stats in current_scenario_points.items():
            final_standings.append({
                'Team': team,
                'Points': stats['Points'],
                'Wins': stats['Wins']
            })

        # Sort by Points, then Wins (basic tie-breaker, ignores NRR)
        final_standings = sorted(final_standings, key=lambda x: (x['Points'], x['Wins']), reverse=True)

        # Get the top 4 teams for this scenario
        top_4_teams = [team_data['Team'] for team_data in final_standings[:4]]

        # Count qualification for each team
        for team in top_4_teams:
            top_4_qualifications[team] += 1

        # Store final points for distribution analysis
        for team in teams:
            total_points_distribution[team].append(current_scenario_points[team]['Points'])

        # ---- PLAYOFF SIMULATION ----
        # Get the top 4 teams with their points
        team1 = {'team': final_standings[0]['Team'], 'points': final_standings[0]['Points']}
        team2 = {'team': final_standings[1]['Team'], 'points': final_standings[1]['Points']}
        team3 = {'team': final_standings[2]['Team'], 'points': final_standings[2]['Points']}
        team4 = {'team': final_standings[3]['Team'], 'points': final_standings[3]['Points']}

        # Track appearances in Qualifier 1 and Eliminator
        qualifier1_appearances[team1['team']] += 1
        qualifier1_appearances[team2['team']] += 1
        eliminator_appearances[team3['team']] += 1
        eliminator_appearances[team4['team']] += 1

        # Simulate Qualifier 1 (1st vs 2nd) with points-based probability
        q1_winner = simulate_playoff_match(
            team1['team'], team2['team'],
            team1['points'], team2['points'],
            bias_factor=0.2
        )
        q1_loser = team2['team'] if q1_winner == team1['team'] else team1['team']

        # Simulate Eliminator (3rd vs 4th) with points-based probability
        eliminator_winner = simulate_playoff_match(
            team3['team'], team4['team'],
            team3['points'],
            team4['points'],
            bias_factor=0.2
        )

        # Track Qualifier 2 appearances
        qualifier2_appearances[q1_loser] += 1
        qualifier2_appearances[eliminator_winner] += 1

        # Simulate Qualifier 2 with points-based probability
        # Use the original points from the league stage
        q1_loser_points = current_scenario_points[q1_loser]['Points']
        eliminator_winner_points = current_scenario_points[eliminator_winner]['Points']

        q2_winner = simulate_playoff_match(
            q1_loser, eliminator_winner,
            q1_loser_points, eliminator_winner_points,
            bias_factor=0.3
        )

        # Track Final appearances
        final_appearances[q1_winner] += 1
        final_appearances[q2_winner] += 1

        # Simulate Final with historical bias
        final_winner = ''
        q1_finalist = q1_winner
        q2_finalist = q2_winner

        q1_finalist_points = current_scenario_points[q1_finalist]['Points']
        q2_finalist_points = current_scenario_points[q2_finalist]['Points']

        # Historical data suggests Qualifier 1 winner has a higher chance (10 out of 14 times)
        q1_win_probability_bias = 10 / 14
        base_probability = 0.5
        biased_probability_q1 = base_probability + (q1_win_probability_bias - base_probability) * 0.6 # Adjust 0.6 for strength of bias

        # Introduce a slight randomness based on points in the final as well
        points_difference_factor = 0.005 # Adjust this to control impact of points difference
        probability_adjustment = (q1_finalist_points - q2_finalist_points) * points_difference_factor
        biased_probability_q1 += probability_adjustment
        biased_probability_q1 = max(0.35, min(0.65, biased_probability_q1)) # Keep within reasonable bounds

        if random.random() < biased_probability_q1:
            final_winner = q1_finalist
        else:
            final_winner = q2_finalist

        # Track tournament winner
        tournament_winners[final_winner] += 1

        # --- Progress Indicator ---
        if (i + 1) % (num_simulations // 10) == 0:
            elapsed = time.time() - start_time
            print(f"  ... simulated {i + 1}/{num_simulations} scenarios ({elapsed:.1f} seconds elapsed)")

    # --- Display Results ---
    end_time = time.time()
    print("-" * 60)
    print(f"Monte Carlo Simulation Complete.")
    print(f"Total simulation time: {end_time - start_time:.2f} seconds for {num_simulations:,} scenarios.")
    print("-" * 60)

    # --- Display Top 4 Qualification Probabilities ---
    print("Estimated Top 4 Qualification Probabilities:")
    results_data = []
    for team in teams:
        count = top_4_qualifications.get(team, 0)
        probability = (count / num_simulations) * 100 if num_simulations > 0 else 0
        results_data.append({
            'Team': team,
            'Top 4 Finishes': f"{count:,}",
            'Qualification %': probability
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by='Qualification %', ascending=False)
    results_df_display = results_df.copy()
    results_df_display['Qualification %'] = results_df_display['Qualification %'].map('{:.2f}%'.format)
    print(results_df_display.to_string(index=False))
    print("\n")

    # --- Display Tournament Winner Probabilities ---
    print("=" * 60)
    print("CHAMPIONSHIP PROBABILITIES (with historical Q1 winner bias):")
    winner_data = []
    for team in teams:
        count = tournament_winners.get(team, 0)
        probability = (count / num_simulations) * 100 if num_simulations > 0 else 0
        winner_data.append({
            'Team': team,
            'Tournament Wins': f"{count:,}",
            'Championship %': probability
        })

    winner_df = pd.DataFrame(winner_data)
    winner_df = winner_df.sort_values(by='Championship %', ascending=False)
    winner_df_display = winner_df.copy()
    winner_df_display['Championship %'] = winner_df_display['Championship %'].map('{:.2f}%'.format)
    print(winner_df_display.to_string(index=False))
    print("=" * 60)

    # --- Display Playoff Stage Advancement Probabilities ---
    print("\nPLAYOFF STAGE ADVANCEMENT PROBABILITIES:")
    playoff_data = []
    for team in teams:
        # Calculate probabilities for each stage
        q1_prob = (qualifier1_appearances.get(team, 0) / num_simulations) * 100
        elim_prob = (eliminator_appearances.get(team, 0) / num_simulations) * 100
        q2_prob = (qualifier2_appearances.get(team, 0) / num_simulations) * 100
        final_prob = (final_appearances.get(team, 0) / num_simulations) * 100
        champ_prob = (tournament_winners.get(team, 0) / num_simulations) * 100

        playoff_data.append({
            'Team': team,
            'Qualifier 1 %': f"{q1_prob:.2f}%",
            'Eliminator %': f"{elim_prob:.2f}%",
            'Qualifier 2 %': f"{q2_prob:.2f}%",
            'Final %': f"{final_prob:.2f}%",
            'Champion %': f"{champ_prob:.2f}%"
        })

    playoff_df = pd.DataFrame(playoff_data)
    # Sort by Champion % (descending)
    playoff_df['Champion % (num)'] = playoff_df['Champion %'].str.rstrip('%').astype(float)
    playoff_df = playoff_df.sort_values(by='Champion % (num)', ascending=False)
    playoff_df = playoff_df.drop(columns=['Champion % (num)'])
    print(playoff_df.to_string(index=False))
    print("=" * 60)

    # --- Display Points Distribution Statistics ---
    print("\nEstimated Final Points Distribution Statistics:")
    points_stats_data = []
    for team in teams:
        pts_list = total_points_distribution.get(team, [])
        if pts_list:
            avg_pts = np.mean(pts_list)
            median_pts = np.median(pts_list)
            mode_pts_series = pd.Series(pts_list).mode()
            mode_pts = ', '.join(map(str, mode_pts_series.tolist())) if not mode_pts_series.empty else 'N/A'
            points_stats_data.append({
                'Team': team,
                'Avg Pts': f"{avg_pts:.1f}",
                'Median Pts': f"{median_pts:.1f}",
                'Mode Pts': mode_pts,
            })
        else:
            points_stats_data.append({
                'Team': team, 'Avg Pts': 'N/A', 'Median Pts': 'N/A', 'Mode Pts': 'N/A', 'Pts Range (25-75%)': 'N/A'
            })

    points_stats_df = pd.DataFrame(points_stats_data)
    # Sort by average points (descending)
    points_stats_df['Avg Pts Num'] = points_stats_df['Avg Pts'].str.replace('N/A', '-1').astype(float)
    points_stats_df = points_stats_df.sort_values(by='Avg Pts Num', ascending=False)
    points_stats_df = points_stats_df.drop(columns=['Avg Pts Num'])
    print(points_stats_df.to_string(index=False))
    print("=" * 60)

    # --- Optional: Create Championship Probability Visualization ---
    plt.figure(figsize=(10, 6))
    teams_sorted = [team for team in winner_df['Team']]
    probs = [prob for prob in winner_df['Championship %']]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.bar(teams_sorted, probs, color=colors[:len(teams_sorted)])
    plt.title('IPL 2025 Championship Probability by Team (with historical bias)', fontsize=14)
    plt.ylabel('Probability (%)')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()

    # Save the figure
    plt.savefig('ipl_2025_championship_probabilities_biased.png', dpi=300)
    print("\nChampionship probability chart (with historical bias) saved as 'ipl_2025_championship_probabilities_biased.png'")

    print("\nNotes:")
    print(" - Probabilities are estimates based on Monte Carlo simulation")
    print(" - League stage uses 50/50 random outcomes for remaining matches")
    print(" - Playoff simulations are weighted by team points (stronger teams have advantage)")
    print(" - The final match simulation now incorporates a historical bias favoring the winner of Qualifier 1.")
    print(" - Results reflect both remaining schedule difficulty and playoff format advantage.")
    print("=" * 60)
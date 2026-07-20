from functools import cmp_to_key

def process_match(standings, wins, team1, team2, score1, score2):
    """Updates the standings and wins based on the match result."""
    # Update the played matches
    standings[team1]["P"] += 1
    standings[team2]["P"] += 1

    # Update goals for and against
    standings[team1]["GF"] += score1
    standings[team1]["GA"] += score2
    standings[team2]["GF"] += score2
    standings[team2]["GA"] += score1
    standings[team1]["GD"] = standings[team1]["GF"] - standings[team1]["GA"]
    standings[team2]["GD"] = standings[team2]["GF"] - standings[team2]["GA"]

    # Update wins, draws, losses, and points
    if score1 > score2:
        standings[team1]["W"] += 1
        standings[team2]["L"] += 1
        standings[team1]["Pts"] += 3
        wins[team1].append(team2)
    elif score1 < score2:
        standings[team2]["W"] += 1
        standings[team1]["L"] += 1
        standings[team2]["Pts"] += 3
        wins[team2].append(team1)
    else:
        standings[team1]["D"] += 1
        standings[team2]["D"] += 1
        standings[team1]["Pts"] += 1
        standings[team2]["Pts"] += 1

def sort_standings(standings, wins):
    """Sorts the standings based on the specified criteria."""
    # This function compares two team names and returns:
    # A positive number if team1 > team2
    # A negative number if team1 < team2
    # 0 if they are exactly tied
    def compare(team1, team2):
        """Compares two teams based on Points, Goal Difference, Head-to-Head, and Goals Scored."""        
        t1_stats = standings[team1]
        t2_stats = standings[team2]

        # 1. Compare Points
        if t1_stats["Pts"] != t2_stats["Pts"]:
            return t1_stats["Pts"] - t2_stats["Pts"]

        # 2. Compare Goal Difference
        if t1_stats["GD"] != t2_stats["GD"]:
            return t1_stats["GD"] - t2_stats["GD"]

        # 3. Head-to-Head (Check this BEFORE Goals Scored)
        if team2 in wins.get(team1, []):
            return 1  # team1 beat team2, so team1 is greater
        elif team1 in wins.get(team2, []):
            return -1 # team2 beat team1, so team2 is greater

        # 4. Fallback to Goals Scored (GF)
        return t1_stats["GF"] - t2_stats["GF"]

    # Get a list of the team names and sort them using our custom compare function
    # reverse=True ensures the "greater" teams end up at the top of the list
    sorted_team_names = sorted(list(standings.keys()), key=cmp_to_key(compare), reverse=True)

    # Rebuild and return the dictionary in the correct sorted order
    return {team: standings[team] for team in sorted_team_names}
    
def print_standings(standings):
    print("\nStandings:")
    print(f"{'Team':<5} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'Pts':<4}")
    for team, stats in standings.items():
        # format GD to show a '+' sign for positive values
        gd_str = f"+{stats['GD']}" if stats['GD'] > 0 else str(stats['GD'])
        print(f"{team:<5} {stats['P']:<3} {stats['W']:<3} {stats['D']:<3} {stats['L']:<3} {stats['GF']:<3} {stats['GA']:<3} {gd_str:<4} {stats['Pts']:<4}")

def main():
    teams = ["ARG", "MEX", "POL", "KSA"]
    keys = ["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]

    standings = {team: {key: 0 for key in keys} for team in teams}

    wins = {team: [] for team in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1 = teams[i]
            team2 = teams[j]
            score = str(input(f"Enter score for {team1} vs {team2} (format: 2-0): "))

            # check if the input is valid
            while(score.count('-') != 1 or not all(part.strip().isdigit() for part in score.split('-'))):
                print("Invalid input format. Please use the format '2-0'.")
                score = str(input(f"Enter score for {team1} vs {team2} (format: 2-0): "))

            goals1, goals2 = map(int, score.split('-'))

            # update standings
            process_match(standings, wins, team1, team2, goals1, goals2)
        
    # print final standings
    print_standings(sort_standings(standings, wins))

if __name__ == "__main__":
    main()
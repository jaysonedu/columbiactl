from api import fetch_player_ratings
from matchup import (
    simulate_match,
    find_best_matchup,
    interactive_game
)

# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if user wants to fetch from API
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Fetch teams from API
        if len(sys.argv) < 4:
            print("Usage: python main.py --api <team1_player1,team1_player2,...> <team2_player1,team2_player2,...>")
            print("Example: python main.py --api player1,player2,player3,player4,player5 opponent1,opponent2,opponent3,opponent4,opponent5")
            sys.exit(1)
        
        team1_players = [p.strip() for p in sys.argv[2].split(",")]
        team2_players = [p.strip() for p in sys.argv[3].split(",")]
        
        if len(team1_players) != 5 or len(team2_players) != 5:
            print("Error: Each team must have exactly 5 players")
            sys.exit(1)
        
        team1 = fetch_player_ratings(team1_players)
        team2 = fetch_player_ratings(team2_players)
        
        # Check if user wants interactive mode
        if len(sys.argv) > 4 and sys.argv[4] == "--interactive":
            interactive_game(team1, team2)
        else:
            # Non-interactive mode with API data
            points1, points2, matchup, t1_banned, t2_banned = find_best_matchup(team1, team2)
            
            print(f"\nTeam 1 players: {team1_players}")
            print(f"Team 1 Glicko-2 ratings: {team1}")
            print(f"\nTeam 2 players: {team2_players}")
            print(f"Team 2 Glicko-2 ratings: {team2}")
            print(f"\nBan Phase:")
            print(f"  Team1 bans Team2 players: {[team2_players[i] for i in t2_banned]}")
            print(f"  Team2 bans Team1 players: {[team1_players[i] for i in t1_banned]}")
            print(f"\nOptimal Matchup Strategy:")
            print("Note: After each match, the winner must send their next player first.")
            print("The matchup shown assumes the most likely outcome at each step.\n")
            for match_num, (t1_idx, t2_idx) in enumerate(matchup, 1):
                r1, rd1 = team1[t1_idx]
                r2, rd2 = team2[t2_idx]
                print(f"Match {match_num}: {team1_players[t1_idx]} (rating {r1:.0f}, RD {rd1:.0f}) vs {team2_players[t2_idx]} (rating {r2:.0f}, RD {rd2:.0f})")
                match_p1, match_p2, win_prob = simulate_match(r1, rd1, r2, rd2)
                print(f"  Expected: {match_p1:.2f} - {match_p2:.2f} (Team1 win prob: {win_prob:.1%})")
            print(f"\nTotal Expected Points:")
            print(f"  Team 1: {points1:.2f}")
            print(f"  Team 2: {points2:.2f}")
            print(f"\nRun with --interactive flag for step-by-step guidance: python main.py --api <teams> --interactive")
    
    else:
        # Example teams: each player is (rating, RD)
        team1 = [(2035, 200), (1955, 200), (1462, 200), (2060, 200), (2882, 200)]
        team2 = [(2387, 200), (2323, 200), (1880, 200), (2248, 200), (2086, 200)]
        
        # Check if user wants interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            interactive_game(team1, team2)
        else:
            # Non-interactive mode: show optimal strategy
            points1, points2, matchup, t1_banned, t2_banned = find_best_matchup(team1, team2)
            
            print(f"Team 1 Glicko-2 ratings: {team1}")
            print(f"Team 2 Glicko-2 ratings: {team2}")
            print(f"\nBan Phase:")
            print(f"  Team1 bans Team2 players at indices: {t2_banned} (ratings: {[team2[i] for i in t2_banned]})")
            print(f"  Team2 bans Team1 players at indices: {t1_banned} (ratings: {[team1[i] for i in t1_banned]})")
            print(f"\nOptimal Matchup Strategy:")
            print("Note: After each match, the winner must send their next player first.")
            print("The matchup shown assumes the most likely outcome at each step.\n")
            for match_num, (t1_idx, t2_idx) in enumerate(matchup, 1):
                r1, rd1 = team1[t1_idx]
                r2, rd2 = team2[t2_idx]
                print(f"Match {match_num}: Team1[{t1_idx}] (rating {r1:.0f}, RD {rd1:.0f}) vs Team2[{t2_idx}] (rating {r2:.0f}, RD {rd2:.0f})")
                match_p1, match_p2, win_prob = simulate_match(r1, rd1, r2, rd2)
                print(f"  Expected: {match_p1:.2f} - {match_p2:.2f} (Team1 win prob: {win_prob:.1%})")
            print(f"\nTotal Expected Points:")
            print(f"  Team 1: {points1:.2f}")
            print(f"  Team 2: {points2:.2f}")
            print(f"\nRun with --api <team1_players> <team2_players> for non-interactive mode or add --interactive flag for step-by-step guidance: python main.py --api <team1_players> <team2_players> --interactive")

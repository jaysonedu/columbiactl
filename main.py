import math
import requests
import time
from typing import List, Tuple, Optional
from urllib.parse import quote

def fetch_player_ratings(players: List[str]) -> List[Tuple[float, float]]:
    """
    Fetch Glicko-2 ratings and RD for a list of players from TETR.IO API.
    
    Args:
        players: List of player usernames or user IDs
        
    Returns:
        List of (rating, RD) tuples in the same order as input players.
        Uses default values (1500, 350) if a player cannot be found or has no league data.
    """
    API_BASE = "https://ch.tetr.io/api/users/"
    results = []
    
    print(f"Fetching ratings for {len(players)} players from TETR.IO API...")
    
    for i, player in enumerate(players):
        try:
            # Rate limiting: wait 1 second between requests (except first)
            if i > 0:
                time.sleep(1)
            
            # Make API request with proper headers (using browser-like User-Agent to avoid 403)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            # URL-encode the player name in case it contains special characters
            # Use lowercase as per API docs: "The lowercase username or user ID to look up"
            encoded_player = quote(player.lower(), safe='')
            
            # Use the league summaries endpoint which has glicko and rd directly
            response = requests.get(f"{API_BASE}{encoded_player}/summaries/league", headers=headers, timeout=5)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success") and "data" in data:
                        league_data = data["data"]
                        
                        # Get glicko and rd directly from data["data"]
                        glicko = league_data.get("glicko")
                        rd = league_data.get("rd")
                        
                        # Handle -1 values (less than 10 games or unranked)
                        if glicko is not None and glicko != -1 and rd is not None and rd != -1:
                            # Check if rd > 100 (unranked per API docs)
                            if rd > 100:
                                print(f"  [{i+1}/{len(players)}] {player}: Unranked (RD > 100: {rd:.1f}), using defaults")
                                results.append((1500.0, 350.0))
                            else:
                                results.append((float(glicko), float(rd)))
                                print(f"  [{i+1}/{len(players)}] {player}: Rating {glicko:.1f}, RD {rd:.1f}")
                        else:
                            # Less than 10 games played or unranked
                            if glicko == -1 or rd == -1:
                                print(f"  [{i+1}/{len(players)}] {player}: Less than 10 games played (unranked), using defaults")
                            else:
                                print(f"  [{i+1}/{len(players)}] {player}: No valid league data, using defaults")
                            results.append((1500.0, 350.0))
                    else:
                        print(f"  [{i+1}/{len(players)}] {player}: API returned unsuccessful response")
                        if "error" in data:
                            error_msg = data["error"]
                            if isinstance(error_msg, dict) and "msg" in error_msg:
                                print(f"    Error: {error_msg['msg']}")
                            else:
                                print(f"    Error: {error_msg}")
                        results.append((1500.0, 350.0))
                except KeyError as e:
                    print(f"  [{i+1}/{len(players)}] {player}: KeyError accessing response data: {e}")
                    if 'data' in locals() and isinstance(data, dict) and "data" in data:
                        print(f"    Data keys: {list(data['data'].keys()) if isinstance(data['data'], dict) else type(data['data'])}")
                    results.append((1500.0, 350.0))
                except Exception as e:
                    print(f"  [{i+1}/{len(players)}] {player}: Error parsing response: {type(e).__name__}: {e}")
                    results.append((1500.0, 350.0))
            elif response.status_code == 403:
                print(f"  [{i+1}/{len(players)}] {player}: HTTP 403 Forbidden - API may require authentication or have rate limiting")
                print(f"    Response: {response.text[:200] if response.text else 'No response body'}")
                results.append((1500.0, 350.0))
            else:
                print(f"  [{i+1}/{len(players)}] {player}: HTTP {response.status_code}, using defaults")
                if response.text:
                    print(f"    Response: {response.text[:200]}")
                results.append((1500.0, 350.0))
                
        except requests.exceptions.RequestException as e:
            print(f"  [{i+1}/{len(players)}] {player}: Error ({str(e)}), using defaults")
            results.append((1500.0, 350.0))
        except Exception as e:
            print(f"  [{i+1}/{len(players)}] {player}: Unexpected error ({str(e)}), using defaults")
            results.append((1500.0, 350.0))
    
    print(f"\nFetched ratings for {len(results)} players.")
    return results


def glicko2_g(phi: float) -> float:
    """Glicko-2 g function: g(φ) = 1 / sqrt(1 + 3φ²/π²)"""
    return 1 / math.sqrt(1 + 3 * phi * phi / (math.pi * math.pi))

def glicko2_expected_score(mu1: float, mu2: float, phi2: float) -> float:
    """
    Calculate expected score for player 1 against player 2 using Glicko-2.
    E(μ, μ_j, φ_j) = 1 / (1 + exp(-g(φ_j)(μ - μ_j)))
    """
    g_phi2 = glicko2_g(phi2)
    return 1 / (1 + math.exp(-g_phi2 * (mu1 - mu2)))

def simulate_match(rating1: float, rd1: float, rating2: float, rd2: float, 
                   sigma1: float = 0.06, sigma2: float = 0.06) -> Tuple[float, float, float]:
    """
    Simulate a first-to-seven game between two players with Glicko-2 ratings.
    rating: player's rating (Glicko scale, typically 1500+)
    rd: rating deviation (typically 30-350)
    sigma: volatility (typically 0.06)
    Returns (expected_points_player1, expected_points_player2, win_probability_player1).
    """
    # Convert to Glicko-2 scale
    mu1 = (rating1 - 1500) / 173.7178
    phi1 = rd1 / 173.7178
    mu2 = (rating2 - 1500) / 173.7178
    phi2 = rd2 / 173.7178
    
    # Calculate win probability using Glicko-2 expected score formula
    win_prob = glicko2_expected_score(mu1, mu2, phi2)
    
    # For a first-to-seven game, calculate expected points
    points1 = 0.0
    points2 = 0.0
    
    # Calculate probability of player1 winning with each score
    for score2 in range(7):
        # Probability of 7 vs score2 (where score2 = 0..6)
        # This is a negative binomial: need 7 wins for player1, score2 wins for player2
        if score2 == 0:
            prob = win_prob ** 7
        else:
            # Number of ways to arrange: 6 wins for p1, score2 wins for p2, then final win for p1
            ways = math.comb(6 + score2, score2)
            prob = ways * (win_prob ** 7) * ((1 - win_prob) ** score2)
        points1 += prob * 7
        points2 += prob * score2
    
    # Calculate probability of player2 winning with each score
    for score1 in range(7):
        # Probability of score1 vs 7 (where score1 = 0..6)
        if score1 == 0:
            prob = (1 - win_prob) ** 7
        else:
            ways = math.comb(6 + score1, score1)
            prob = ways * ((1 - win_prob) ** 7) * (win_prob ** score1)
        points1 += prob * score1
        points2 += prob * 7
    
    return (points1, points2, win_prob)


def calculate(t1: List[Tuple[float, float]], t2: List[Tuple[float, float]], must_send_first: int = None, 
              t1_indices: List[int] = None, t2_indices: List[int] = None,
              t1_counterpick_used: bool = False, t2_counterpick_used: bool = False) -> Tuple[float, float, List[Tuple[int, int]]]:
    """
    Find the optimal matchup between two teams to maximize t1's points.
    must_send_first: 1 if team1 must send first, 2 if team2 must send first, None if first match
    t1_indices/t2_indices: lists of original indices corresponding to current players in t1/t2
    t1_counterpick_used/t2_counterpick_used: whether each team has used their counterpick
    Returns (best_t1_points, best_t2_points, best_matchup).
    best_matchup is a list of (t1_player_index, t2_player_index) pairs.
    """
    # Initialize indices if not provided (each player maps to its position)
    if t1_indices is None:
        t1_indices = list(range(len(t1)))
    if t2_indices is None:
        t2_indices = list(range(len(t2)))
    
    if len(t1) == 0 and len(t2) == 0:
        return (0.0, 0.0, [])

    if len(t1) == 1 and len(t2) == 1:
        r1, rd1 = t1[0]
        r2, rd2 = t2[0]
        points1, points2, _ = simulate_match(r1, rd1, r2, rd2)
        return (points1, points2, [(t1_indices[0], t2_indices[0])])
    
    best_points1 = -1.0
    best_points2 = 0.0
    best_matchup = []
    
    if must_send_first == 1:
        # Team1 must send first, team2 chooses optimally
        for i, player1 in enumerate(t1):
            best_response_points1 = float('inf')
            best_response_points2 = -1.0
            best_response_matchup = []
            
            # Team2 chooses best response
            for j, player2 in enumerate(t2):
                # Simulate this match
                r1, rd1 = player1
                r2, rd2 = player2
                match_points1, match_points2, win_prob1 = simulate_match(r1, rd1, r2, rd2)
                
                # Create new teams without the matched players, and track their indices
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                new_t1_indices = t1_indices[:i] + t1_indices[i+1:]
                new_t2_indices = t2_indices[:j] + t2_indices[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    if 7.0 + keep_win_p2 > 6.0 + counterpick_p2 or (7.0 + keep_win_p1 < 6.0 + counterpick_p1):
                        if_win2_points1, if_win2_points2, if_win2_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win2_points2 = 7.0 + if_win2_points2
                    else:
                        if_win2_points1, if_win2_points2, if_win2_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win2_points2 = 6.0 + if_win2_points2
                
                # Expected value considering win probabilities
                total_points1 = win_prob1 * win1_points1 + (1 - win_prob1) * if_win2_points1
                total_points2 = win_prob1 * if_win1_points2 + (1 - win_prob1) * win2_points2
                
                # Team2 minimizes team1's points (maximizes team2's points)
                if total_points1 < best_response_points1 or (total_points1 == best_response_points1 and total_points2 > best_response_points2):
                    best_response_points1 = total_points1
                    best_response_points2 = total_points2
                    t1_idx = t1_indices[i]
                    t2_idx = t2_indices[j]
                    # Store both possible matchups - we'll use the more likely one for display
                    # In practice, the actual matchup depends on who wins
                    if win_prob1 >= 0.5:  # More likely team1 wins
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win1_matchup
                    else:
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win2_matchup
            
            # Team1 chooses the best option given team2's response
            if best_response_points1 > best_points1:
                best_points1 = best_response_points1
                best_points2 = best_response_points2
                best_matchup = best_response_matchup
                
    elif must_send_first == 2:
        # Team2 must send first, team1 chooses optimally
        for j, player2 in enumerate(t2):
            best_response_points1 = -1.0
            best_response_points2 = 0.0
            best_response_matchup = []
            
            # Team1 chooses best response
            for i, player1 in enumerate(t1):
                # Simulate this match
                r1, rd1 = player1
                r2, rd2 = player2
                match_points1, match_points2, win_prob1 = simulate_match(r1, rd1, r2, rd2)
                
                # Create new teams without the matched players, and track their indices
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                new_t1_indices = t1_indices[:i] + t1_indices[i+1:]
                new_t2_indices = t2_indices[:j] + t2_indices[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    if 7.0 + keep_win_p2 > 6.0 + counterpick_p2 or (7.0 + keep_win_p1 < 6.0 + counterpick_p1):
                        if_win2_points1, if_win2_points2, if_win2_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win2_points2 = 7.0 + if_win2_points2
                    else:
                        if_win2_points1, if_win2_points2, if_win2_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win2_points2 = 6.0 + if_win2_points2
                
                # Expected value considering win probabilities
                total_points1 = win_prob1 * win1_points1 + (1 - win_prob1) * if_win2_points1
                total_points2 = win_prob1 * if_win1_points2 + (1 - win_prob1) * win2_points2
                
                # Team1 maximizes team1's points
                if total_points1 > best_response_points1:
                    best_response_points1 = total_points1
                    best_response_points2 = total_points2
                    t1_idx = t1_indices[i]
                    t2_idx = t2_indices[j]
                    # Store both possible matchups - we'll use the more likely one for display
                    if win_prob1 >= 0.5:  # More likely team1 wins
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win1_matchup
                    else:
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win2_matchup
            
            # Team2 chooses the best option given team1's response
            if best_response_points1 > best_points1:
                best_points1 = best_response_points1
                best_points2 = best_response_points2
                best_matchup = best_response_matchup
                
    else:
        # First match - both teams choose optimally
        # Team1 chooses first player, team2 responds optimally
        for i, player1 in enumerate(t1):
            best_response_points1 = float('inf')
            best_response_points2 = -1.0
            best_response_matchup = []
            
            # Team2 chooses best response
            for j, player2 in enumerate(t2):
                # Simulate this match
                r1, rd1 = player1
                r2, rd2 = player2
                match_points1, match_points2, win_prob1 = simulate_match(r1, rd1, r2, rd2)
                
                # Create new teams without the matched players, and track their indices
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                new_t1_indices = t1_indices[:i] + t1_indices[i+1:]
                new_t2_indices = t2_indices[:j] + t2_indices[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, t1_counterpick_used, True)
                    if 7.0 + keep_win_p2 > 6.0 + counterpick_p2 or (7.0 + keep_win_p1 < 6.0 + counterpick_p1):
                        if_win2_points1, if_win2_points2, if_win2_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win2_points2 = 7.0 + if_win2_points2
                    else:
                        if_win2_points1, if_win2_points2, if_win2_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win2_points2 = 6.0 + if_win2_points2
                
                # Expected value considering win probabilities
                total_points1 = win_prob1 * win1_points1 + (1 - win_prob1) * if_win2_points1
                total_points2 = win_prob1 * if_win1_points2 + (1 - win_prob1) * win2_points2
                
                # Team2 minimizes team1's points
                if total_points1 < best_response_points1 or (total_points1 == best_response_points1 and total_points2 > best_response_points2):
                    best_response_points1 = total_points1
                    best_response_points2 = total_points2
                    t1_idx = t1_indices[i]
                    t2_idx = t2_indices[j]
                    # Store both possible matchups - we'll use the more likely one for display
                    if win_prob1 >= 0.5:  # More likely team1 wins
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win1_matchup
                    else:
                        best_response_matchup = [(t1_idx, t2_idx)] + if_win2_matchup
            
            # Team1 chooses the best option given team2's response
            if best_response_points1 > best_points1:
                best_points1 = best_response_points1
                best_points2 = best_response_points2
                best_matchup = best_response_matchup
    
    return (best_points1, best_points2, best_matchup)


def find_optimal_bans(t1: List[Tuple[float, float]], t2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int], List[int]]:
    """
    Find optimal bans: each team bans 2 players from opposing team in alternating rounds.
    Bans only apply to the first match; banned players are available for subsequent matches.
    Round 1: Team1 bans, then Team2 bans. Round 2: Team1 bans, then Team2 bans.
    Returns (t1_for_match1, t2_for_match1, t1_banned_indices, t2_banned_indices).
    """
    best_t1_points = -1.0
    best_t1_for_match1 = []
    best_t2_for_match1 = []
    best_t1_banned = []
    best_t2_banned = []
    
    # Round 1: Team1 bans first
    for ban1_idx in range(len(t2)):
        t2_after_r1_ban1 = t2[:ban1_idx] + t2[ban1_idx+1:]
        
        # Round 1: Team2 bans (minimizes team1's expected points)
        best_r1_ban2_points = float('inf')
        best_r1_ban2_idx = -1
        
        for ban2_idx in range(len(t1)):
            t1_after_r1 = t1[:ban2_idx] + t1[ban2_idx+1:]
            
            # Round 2: Team1 bans
            best_r2_ban1_points = -1.0
            best_r2_ban1_idx = -1
            
            for ban3_idx in range(len(t2_after_r1_ban1)):
                t2_after_r2_ban1 = t2_after_r1_ban1[:ban3_idx] + t2_after_r1_ban1[ban3_idx+1:]
                
                # Round 2: Team2 bans (minimizes team1's expected points)
                best_r2_ban2_points = float('inf')
                best_r2_ban2_idx = -1
                
                for ban4_idx in range(len(t1_after_r1)):
                    t1_for_match1 = t1_after_r1[:ban4_idx] + t1_after_r1[ban4_idx+1:]
                    # Get original index of second banned player
                    ban4_original_idx = t1.index(t1_after_r1[ban4_idx])
                    # Simple heuristic: find best match 1 pairing
                    best_match1_score = -1.0
                    for p1 in t1_for_match1:
                        for p2 in t2_after_r2_ban1:
                            r1, rd1 = p1
                            r2, rd2 = p2
                            m1_p1, _, _ = simulate_match(r1, rd1, r2, rd2)
                            if m1_p1 > best_match1_score:
                                best_match1_score = m1_p1
                    # Use match 1 score as heuristic (full optimization happens in find_best_matchup)
                    if best_match1_score < best_r2_ban2_points:
                        best_r2_ban2_points = best_match1_score
                        best_r2_ban2_idx = ban4_idx
                        best_r2_ban2_original_idx = ban4_original_idx
                
                if best_r2_ban2_points > best_r2_ban1_points:
                    best_r2_ban1_points = best_r2_ban2_points
                    best_r2_ban1_idx = ban3_idx
            
            # Team2's round 1 choice: minimize team1's points after round 2
            if best_r2_ban1_idx >= 0:
                t1_for_match1 = t1_after_r1[:best_r2_ban2_idx] + t1_after_r1[best_r2_ban2_idx+1:]
                t2_for_match1 = t2_after_r1_ban1[:best_r2_ban1_idx] + t2_after_r1_ban1[best_r2_ban1_idx+1:]
                # Get original index of second t2 banned player
                ban3_original_idx = t2.index(t2_after_r1_ban1[best_r2_ban1_idx])
                # Simple heuristic: best match 1 score
                best_match1_score = -1.0
                for p1 in t1_for_match1:
                    for p2 in t2_for_match1:
                        r1, rd1 = p1
                        r2, rd2 = p2
                        m1_p1, _, _ = simulate_match(r1, rd1, r2, rd2)
                        if m1_p1 > best_match1_score:
                            best_match1_score = m1_p1
                if best_match1_score < best_r1_ban2_points:
                    best_r1_ban2_points = best_match1_score
                    best_r1_ban2_idx = ban2_idx
                    best_r2_ban2_original_idx_saved = best_r2_ban2_original_idx
                    best_r2_ban1_original_idx_saved = ban3_original_idx
        
        # Team1's round 1 choice: maximize points after team2's optimal response
        if best_r1_ban2_idx >= 0:
            t1_after_r1 = t1[:best_r1_ban2_idx] + t1[best_r1_ban2_idx+1:]
            # Recalculate round 2 with optimal round 1 choices
            best_r2_ban1_points = -1.0
            best_r2_ban1_idx = -1
            best_r2_ban2_idx = -1
            
            for ban3_idx in range(len(t2_after_r1_ban1)):
                t2_after_r2_ban1 = t2_after_r1_ban1[:ban3_idx] + t2_after_r1_ban1[ban3_idx+1:]
                best_r2_ban2_points_temp = float('inf')
                best_r2_ban2_idx_temp = -1
                
                for ban4_idx in range(len(t1_after_r1)):
                    t1_for_match1 = t1_after_r1[:ban4_idx] + t1_after_r1[ban4_idx+1:]
                    ban4_original_idx = t1.index(t1_after_r1[ban4_idx])
                    ban3_original_idx = t2.index(t2_after_r1_ban1[ban3_idx])
                    # Simple heuristic: best match 1 score
                    best_match1_score = -1.0
                    for p1 in t1_for_match1:
                        for p2 in t2_after_r2_ban1:
                            r1, rd1 = p1
                            r2, rd2 = p2
                            m1_p1, _, _ = simulate_match(r1, rd1, r2, rd2)
                            if m1_p1 > best_match1_score:
                                best_match1_score = m1_p1
                    if best_match1_score < best_r2_ban2_points_temp:
                        best_r2_ban2_points_temp = best_match1_score
                        best_r2_ban2_idx_temp = ban4_idx
                        best_r2_ban2_original_idx_temp = ban4_original_idx
                        best_r2_ban1_original_idx_temp = ban3_original_idx
                
                if best_r2_ban2_points_temp > best_r2_ban1_points:
                    best_r2_ban1_points = best_r2_ban2_points_temp
                    best_r2_ban1_idx = ban3_idx
                    best_r2_ban2_idx = best_r2_ban2_idx_temp
            
            if best_r2_ban1_idx >= 0 and best_r2_ban2_idx >= 0:
                t1_for_match1 = t1_after_r1[:best_r2_ban2_idx] + t1_after_r1[best_r2_ban2_idx+1:]
                t2_for_match1 = t2_after_r1_ban1[:best_r2_ban1_idx] + t2_after_r1_ban1[best_r2_ban1_idx+1:]
                # Simple heuristic: best match 1 score
                best_match1_score = -1.0
                for p1 in t1_for_match1:
                    for p2 in t2_for_match1:
                        r1, rd1 = p1
                        r2, rd2 = p2
                        m1_p1, _, _ = simulate_match(r1, rd1, r2, rd2)
                        if m1_p1 > best_match1_score:
                            best_match1_score = m1_p1
                if best_match1_score > best_t1_points:
                    best_t1_points = best_match1_score
                    best_t1_for_match1 = t1_for_match1
                    best_t2_for_match1 = t2_for_match1
                    best_t1_banned = [best_r1_ban2_idx, best_r2_ban2_original_idx_temp]
                    best_t2_banned = [ban1_idx, best_r2_ban1_original_idx_temp]
    
    return (best_t1_for_match1, best_t2_for_match1, best_t1_banned, best_t2_banned)


def find_best_matchup(t1: List[Tuple[float, float]], t2: List[Tuple[float, float]]) -> Tuple[float, float, List[Tuple[int, int]], List[int], List[int]]:
    """
    Wrapper function to find the best matchup between two teams with bans.
    Bans only apply to the first match; banned players are available for subsequent matches.
    Returns (t1_total_points, t2_total_points, matchup, t1_banned_indices, t2_banned_indices).
    matchup is a list of (t1_index, t2_index) pairs showing optimal pairing.
    """
    t1_for_match1, t2_for_match1, t1_banned, t2_banned = find_optimal_bans(t1, t2)
    
    # Calculate first match with banned players removed
    # Then calculate remaining matches with all players (including banned ones) available
    best_total_points1 = -1.0
    best_total_points2 = 0.0
    best_matchup = []
    
    # Try all possible first match pairings with banned players removed
    for i, player1 in enumerate(t1_for_match1):
        for j, player2 in enumerate(t2_for_match1):
            r1, rd1 = player1
            r2, rd2 = player2
            match1_points1, match1_points2, win_prob1 = simulate_match(r1, rd1, r2, rd2)
            
            # Create teams for remaining matches (add back banned players, exclude match 1 players)
            t1_remaining = t1_for_match1[:i] + t1_for_match1[i+1:] + [t1[idx] for idx in t1_banned]
            t2_remaining = t2_for_match1[:j] + t2_for_match1[j+1:] + [t2[idx] for idx in t2_banned]
            
            # Track indices for remaining players
            # Compute indices: t1_for_match1 contains players NOT in t1_banned
            # So indices are all indices except banned ones
            t1_all_indices = list(range(len(t1)))
            t2_all_indices = list(range(len(t2)))
            t1_for_match1_indices = [idx for idx in t1_all_indices if idx not in t1_banned]
            t2_for_match1_indices = [idx for idx in t2_all_indices if idx not in t2_banned]
            # Remove match 1 player indices and add back banned indices
            t1_match1_idx = t1_for_match1_indices[i]
            t2_match1_idx = t2_for_match1_indices[j]
            # Ensure indices are unique and match the players list
            t1_remaining_indices = [idx for idx in t1_for_match1_indices if idx != t1_match1_idx] + t1_banned
            t2_remaining_indices = [idx for idx in t2_for_match1_indices if idx != t2_match1_idx] + t2_banned
            # Remove any duplicates (shouldn't happen, but safety check)
            t1_remaining_indices = list(dict.fromkeys(t1_remaining_indices))  # Preserves order
            t2_remaining_indices = list(dict.fromkeys(t2_remaining_indices))
            
            # If team1 wins match 1, team1 must send first next
            # If team2 wins match 1, team2 must send first next
            if_win1_points1, if_win1_points2, if_win1_matchup = calculate(t1_remaining, t2_remaining, 1, t1_remaining_indices, t2_remaining_indices)
            if_win2_points1, if_win2_points2, if_win2_matchup = calculate(t1_remaining, t2_remaining, 2, t1_remaining_indices, t2_remaining_indices)
            
            # Expected value considering win probabilities
            total_points1 = match1_points1 + win_prob1 * if_win1_points1 + (1 - win_prob1) * if_win2_points1
            total_points2 = match1_points2 + win_prob1 * if_win1_points2 + (1 - win_prob1) * if_win2_points2
            
            # Find original indices for first match (already computed above)
            t1_idx = t1_match1_idx
            t2_idx = t2_match1_idx
            
            if total_points1 > best_total_points1:
                best_total_points1 = total_points1
                best_total_points2 = total_points2
                # Use the more likely outcome for matchup display
                if win_prob1 >= 0.5:
                    best_matchup = [(t1_idx, t2_idx)] + if_win1_matchup
                else:
                    best_matchup = [(t1_idx, t2_idx)] + if_win2_matchup
    
    return (best_total_points1, best_total_points2, best_matchup, t1_banned, t2_banned)


def get_optimal_next_match(t1: List[Tuple[float, float]], t2: List[Tuple[float, float]], 
                           t1_played: List[int], t2_played: List[int],
                           must_send_first: int = None, t1_counterpick_used: bool = False, 
                           t2_counterpick_used: bool = False, t1_banned: List[int] = None,
                           t2_banned: List[int] = None) -> Tuple[int, int, float, float, List[Tuple[int, int]]]:
    """
    Get optimal next match given current state.
    t1_played/t2_played: indices of players who have already played
    t1_banned/t2_banned: indices of banned players (only used for match 1)
    Returns (t1_player_idx, t2_player_idx, expected_t1_points, expected_t2_points, remaining_matchup).
    """
    # Get remaining players (exclude played and banned)
    if t1_banned is None:
        t1_banned = []
    if t2_banned is None:
        t2_banned = []
    
    # t1_banned = team1 players banned by team2, t2_banned = team2 players banned by team1
    t1_remaining = [t1[i] for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
    t2_remaining = [t2[i] for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
    t1_remaining_indices = [i for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
    t2_remaining_indices = [i for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
    
    if len(t1_remaining) == 0 or len(t2_remaining) == 0:
        return (-1, -1, 0.0, 0.0, [])
    
    # Find optimal matchup
    points1, points2, matchup = calculate(t1_remaining, t2_remaining, must_send_first, 
                                           t1_remaining_indices, t2_remaining_indices,
                                           t1_counterpick_used, t2_counterpick_used)
    
    if len(matchup) == 0:
        return (-1, -1, 0.0, 0.0, [])
    
    # First match in the matchup is the optimal next match
    next_t1_idx, next_t2_idx = matchup[0]
    remaining_matchup = matchup[1:]
    
    return (next_t1_idx, next_t2_idx, points1, points2, remaining_matchup)


def interactive_game(t1: List[Tuple[float, float]], t2: List[Tuple[float, float]]):
    """
    Interactive function to play through a game, updating at each phase.
    """
    print("=" * 60)
    print("Interactive Tetris Team Match Calculator")
    print("=" * 60)
    print(f"\nTeam 1: {len(t1)} players")
    for i, (r, rd) in enumerate(t1):
        print(f"  [{i}] Rating: {r:.0f}, RD: {rd:.0f}")
    print(f"\nTeam 2: {len(t2)} players")
    for i, (r, rd) in enumerate(t2):
        print(f"  [{i}] Rating: {r:.0f}, RD: {rd:.0f}")
    
    # Phase 1: Bans (Two rounds)
    print("\n" + "=" * 60)
    print("PHASE 1: BAN SELECTION - ROUND 1")
    print("=" * 60)
    
    # Get optimal bans for reference
    _, _, optimal_t1_banned, optimal_t2_banned = find_optimal_bans(t1, t2)
    
    # Round 1: Team1 bans first, then Team2 bans
    print("\nRound 1 - Team1 bans first:")
    print(f"  Optimal: Team1 should ban Team2 player: {optimal_t2_banned[0] if optimal_t2_banned else 'N/A'} {t2[optimal_t2_banned[0]] if optimal_t2_banned else ''}")
    t2_available_r1 = [i for i in range(len(t2))]
    print(f"  Available Team2 players: {t2_available_r1}")
    ban1_input = input("  Team1 bans Team2 player (index, or Enter for optimal): ").strip()
    ban1 = optimal_t2_banned[0] if not ban1_input and optimal_t2_banned else (int(ban1_input) if ban1_input else -1)
    
    print(f"\nRound 1 - Team2 bans:")
    print(f"  Optimal: Team2 should ban Team1 player: {optimal_t1_banned[0] if optimal_t1_banned else 'N/A'} {t1[optimal_t1_banned[0]] if optimal_t1_banned else ''}")
    t1_available_r1 = [i for i in range(len(t1))]
    print(f"  Available Team1 players: {t1_available_r1}")
    ban2_input = input("  Team2 bans Team1 player (index, or Enter for optimal): ").strip()
    ban2 = optimal_t1_banned[0] if not ban2_input and optimal_t1_banned else (int(ban2_input) if ban2_input else -1)
    
    print("\n" + "=" * 60)
    print("PHASE 1: BAN SELECTION - ROUND 2")
    print("=" * 60)
    
    # Round 2: Team1 bans second, then Team2 bans second
    print("\nRound 2 - Team1 bans second:")
    print(f"  Optimal: Team1 should ban Team2 player: {optimal_t2_banned[1] if len(optimal_t2_banned) > 1 else 'N/A'} {t2[optimal_t2_banned[1]] if len(optimal_t2_banned) > 1 else ''}")
    t2_available_r2 = [i for i in range(len(t2)) if i != ban1]
    print(f"  Available Team2 players: {t2_available_r2}")
    ban3_input = input("  Team1 bans Team2 player (index, or Enter for optimal): ").strip()
    ban3 = optimal_t2_banned[1] if not ban3_input and len(optimal_t2_banned) > 1 else (int(ban3_input) if ban3_input else -1)
    
    print(f"\nRound 2 - Team2 bans second:")
    print(f"  Optimal: Team2 should ban Team1 player: {optimal_t1_banned[1] if len(optimal_t1_banned) > 1 else 'N/A'} {t1[optimal_t1_banned[1]] if len(optimal_t1_banned) > 1 else ''}")
    t1_available_r2 = [i for i in range(len(t1)) if i != ban2]
    print(f"  Available Team1 players: {t1_available_r2}")
    ban4_input = input("  Team2 bans Team1 player (index, or Enter for optimal): ").strip()
    ban4 = optimal_t1_banned[1] if not ban4_input and len(optimal_t1_banned) > 1 else (int(ban4_input) if ban4_input else -1)
    
    # Store final bans
    t1_banned = [ban2, ban4]  # Team2 banned these Team1 players
    t2_banned = [ban1, ban3]  # Team1 banned these Team2 players
    
    print(f"\nFinal bans:")
    print(f"  Team1 banned Team2 players: {t2_banned} {[t2[i] for i in t2_banned]}")
    print(f"  Team2 banned Team1 players: {t1_banned} {[t1[i] for i in t1_banned]}")
    
    # Initialize game state
    t1_played = []
    t2_played = []
    t1_score = 0.0
    t2_score = 0.0
    t1_counterpick_used = False
    t2_counterpick_used = False
    must_send_first = None  # None means first match, no restriction
    match_num = 1
    
    # Phase 2: Matches
    while len(t1_played) < len(t1) and len(t2_played) < len(t2):
        print("\n" + "=" * 60)
        print(f"PHASE {match_num + 1}: MATCH {match_num}")
        print("=" * 60)
        
        # Get optimal next match
        if match_num == 1:
            # Match 1: exclude banned players
            next_t1_idx, next_t2_idx, exp_points1, exp_points2, remaining_matchup = get_optimal_next_match(
                t1, t2, [], [], must_send_first, t1_counterpick_used, t2_counterpick_used, t1_banned, t2_banned
            )
        else:
            # Subsequent matches: banned players are now available
            next_t1_idx, next_t2_idx, exp_points1, exp_points2, remaining_matchup = get_optimal_next_match(
                t1, t2, t1_played, t2_played, must_send_first, t1_counterpick_used, t2_counterpick_used
            )
        
        if next_t1_idx == -1:
            print("No more matches possible!")
            break
        
        # Get actual match result based on who must send first
        print("\nEnter match result:")
        actual_t1_idx = -1
        actual_t2_idx = -1
        
        if must_send_first == 1:
            # Team1 must send first
            print(f"  Team1 must send first (winner of previous match)")
            print(f"  Optimal: Team1 should send [{next_t1_idx}]")
            t1_player_input = input(f"  Team1 player index (or Enter for {next_t1_idx}): ").strip()
            actual_t1_idx = next_t1_idx if not t1_player_input else int(t1_player_input)
            
            # Calculate Team2's optimal response to Team1's choice
            if match_num == 1:
                t1_remaining = [t1[i] for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
                t2_remaining = [t2[i] for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
                t1_remaining_indices = [i for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
                t2_remaining_indices = [i for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
            else:
                t1_remaining = [t1[i] for i in range(len(t1)) if i not in t1_played]
                t2_remaining = [t2[i] for i in range(len(t2)) if i not in t2_played]
                t1_remaining_indices = [i for i in range(len(t1)) if i not in t1_played]
                t2_remaining_indices = [i for i in range(len(t2)) if i not in t2_played]
            
            # Find optimal Team2 response
            best_t2_response = -1
            best_t2_points1 = float('inf')
            for j, player2 in enumerate(t2_remaining):
                r1, rd1 = t1[actual_t1_idx]
                r2, rd2 = player2
                match_p1, match_p2, _ = simulate_match(r1, rd1, r2, rd2)
                new_t1 = [p for p in t1_remaining if p != t1[actual_t1_idx]]
                new_t2 = t2_remaining[:j] + t2_remaining[j+1:]
                new_t1_indices = [idx for idx in t1_remaining_indices if idx != actual_t1_idx]
                new_t2_indices = t2_remaining_indices[:j] + t2_remaining_indices[j+1:]
                remaining_p1, _, _ = calculate(new_t1, new_t2, 2, new_t1_indices, new_t2_indices, t1_counterpick_used, t2_counterpick_used)
                total_p1 = match_p1 + remaining_p1
                if total_p1 < best_t2_points1:
                    best_t2_points1 = total_p1
                    best_t2_response = t2_remaining_indices[j]
            
            print(f"  Optimal Team2 response: [{best_t2_response}]")
            t2_player_input = input(f"  Team2 player index (or Enter for {best_t2_response}): ").strip()
            actual_t2_idx = best_t2_response if not t2_player_input else int(t2_player_input)
            
        elif must_send_first == 2:
            # Team2 must send first
            print(f"  Team2 must send first (winner of previous match)")
            print(f"  Optimal: Team2 should send [{next_t2_idx}]")
            t2_player_input = input(f"  Team2 player index (or Enter for {next_t2_idx}): ").strip()
            actual_t2_idx = next_t2_idx if not t2_player_input else int(t2_player_input)
            
            # Calculate Team1's optimal response to Team2's choice
            if match_num == 1:
                t1_remaining = [t1[i] for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
                t2_remaining = [t2[i] for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
                t1_remaining_indices = [i for i in range(len(t1)) if i not in t1_played and i not in t1_banned]
                t2_remaining_indices = [i for i in range(len(t2)) if i not in t2_played and i not in t2_banned]
            else:
                t1_remaining = [t1[i] for i in range(len(t1)) if i not in t1_played]
                t2_remaining = [t2[i] for i in range(len(t2)) if i not in t2_played]
                t1_remaining_indices = [i for i in range(len(t1)) if i not in t1_played]
                t2_remaining_indices = [i for i in range(len(t2)) if i not in t2_played]
            
            # Find optimal Team1 response
            best_t1_response = -1
            best_t1_points1 = -1.0
            for i, player1 in enumerate(t1_remaining):
                r1, rd1 = player1
                r2, rd2 = t2[actual_t2_idx]
                match_p1, match_p2, _ = simulate_match(r1, rd1, r2, rd2)
                new_t1 = t1_remaining[:i] + t1_remaining[i+1:]
                new_t2 = [p for p in t2_remaining if p != t2[actual_t2_idx]]
                new_t1_indices = t1_remaining_indices[:i] + t1_remaining_indices[i+1:]
                new_t2_indices = [idx for idx in t2_remaining_indices if idx != actual_t2_idx]
                remaining_p1, _, _ = calculate(new_t1, new_t2, 1, new_t1_indices, new_t2_indices, t1_counterpick_used, t2_counterpick_used)
                total_p1 = match_p1 + remaining_p1
                if total_p1 > best_t1_points1:
                    best_t1_points1 = total_p1
                    best_t1_response = t1_remaining_indices[i]
            
            print(f"  Optimal Team1 response: [{best_t1_response}]")
            t1_player_input = input(f"  Team1 player index (or Enter for {best_t1_response}): ").strip()
            actual_t1_idx = best_t1_response if not t1_player_input else int(t1_player_input)
            
        else:
            # First match - show full optimal matchup
            print(f"\nOptimal next match:")
            r1, rd1 = t1[next_t1_idx]
            r2, rd2 = t2[next_t2_idx]
            print(f"  Team1[{next_t1_idx}] (rating {r1:.0f}, RD {rd1:.0f}) vs Team2[{next_t2_idx}] (rating {r2:.0f}, RD {rd2:.0f})")
            match_p1, match_p2, win_prob = simulate_match(r1, rd1, r2, rd2)
            print(f"  Expected: {match_p1:.2f} - {match_p2:.2f} (Team1 win prob: {win_prob:.1%})")
            
            t1_player_input = input(f"  Team1 player index (or Enter for {next_t1_idx}): ").strip()
            t2_player_input = input(f"  Team2 player index (or Enter for {next_t2_idx}): ").strip()
            
            actual_t1_idx = next_t1_idx if not t1_player_input else int(t1_player_input)
            actual_t2_idx = next_t2_idx if not t2_player_input else int(t2_player_input)
        
        # Get match score
        score_input = input("  Match score (format: '7-4' or '6-7' for counterpick): ").strip()
        t1_match_score, t2_match_score = map(int, score_input.split("-"))
        
        # Check for counterpick
        used_counterpick = False
        if (t1_match_score == 6 and t2_match_score == 7) or (t1_match_score == 7 and t2_match_score == 6):
            counterpick_input = input("  Was this a counterpick? (y/n): ").strip().lower()
            if counterpick_input == 'y':
                used_counterpick = True
                if t1_match_score == 6:  # Team1 counterpicked (took 6-7 loss)
                    t1_counterpick_used = True
                    t1_match_score, t2_match_score = 6, 7
                else:  # Team2 counterpicked
                    t2_counterpick_used = True
                    t1_match_score, t2_match_score = 7, 6
        
        # Update scores
        t1_score += t1_match_score
        t2_score += t2_match_score
        
        # Update played players
        t1_played.append(actual_t1_idx)
        t2_played.append(actual_t2_idx)
        
        # Determine who must send first next
        if t1_match_score == 7:
            must_send_first = 1  # Team1 won, must send first
        elif t2_match_score == 7:
            must_send_first = 2  # Team2 won, must send first
        else:
            # Counterpick case - winner is the one who got 7
            if t1_match_score == 6 and t2_match_score == 7:
                must_send_first = 2  # Team2 won (even though team1 counterpicked)
            else:
                must_send_first = 1  # Team1 won
        
        # Display match result with winner's player first
        if t1_match_score == 7:
            # Team1 won - show Team1 first
            print(f"\nMatch {match_num} result: Team1[{actual_t1_idx}] {t1_match_score} - {t2_match_score} Team2[{actual_t2_idx}]")
        else:
            # Team2 won - show Team2 first
            print(f"\nMatch {match_num} result: Team2[{actual_t2_idx}] {t2_match_score} - {t1_match_score} Team1[{actual_t1_idx}]")
        print(f"Current score: Team1 {t1_score:.0f} - {t2_score:.0f} Team2")
        print(f"Remaining players:")
        print(f"  Team1: {[i for i in range(len(t1)) if i not in t1_played]}")
        print(f"  Team2: {[i for i in range(len(t2)) if i not in t2_played]}")
        
        match_num += 1
    
    print("\n" + "=" * 60)
    print("GAME COMPLETE")
    print("=" * 60)
    print(f"Final score: Team1 {t1_score:.0f} - {t2_score:.0f} Team2")


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
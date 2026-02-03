import math
from typing import List, Tuple

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
              t1_original: List[int] = None, t2_original: List[int] = None,
              t1_counterpick_used: bool = False, t2_counterpick_used: bool = False) -> Tuple[float, float, List[Tuple[int, int]]]:
    """
    Find the optimal matchup between two teams to maximize t1's points.
    must_send_first: 1 if team1 must send first, 2 if team2 must send first, None if first match
    t1_counterpick_used/t2_counterpick_used: whether each team has used their counterpick
    Returns (best_t1_points, best_t2_points, best_matchup).
    best_matchup is a list of (t1_player_index, t2_player_index) pairs.
    """
    # Store original teams for index mapping
    if t1_original is None:
        t1_original = t1.copy()
    if t2_original is None:
        t2_original = t2.copy()
    
    if len(t1) == 0 and len(t2) == 0:
        return (0.0, 0.0, [])

    if len(t1) == 1 and len(t2) == 1:
        r1, rd1 = t1[0]
        r2, rd2 = t2[0]
        points1, points2, _ = simulate_match(r1, rd1, r2, rd2)
        t1_idx = t1_original.index(t1[0])
        t2_idx = t2_original.index(t2[0])
        return (points1, points2, [(t1_idx, t2_idx)])
    
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
                
                # Create new teams without the matched players
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, t1_counterpick_used, True)
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
                    t1_idx = t1_original.index(player1)
                    t2_idx = t2_original.index(player2)
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
                
                # Create new teams without the matched players
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, t1_counterpick_used, True)
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
                    t1_idx = t1_original.index(player1)
                    t2_idx = t2_original.index(player2)
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
                
                # Create new teams without the matched players
                new_t1 = t1[:i] + t1[i+1:]
                new_t2 = t2[:j] + t2[j+1:]
                
                # If team1 wins, consider counterpick option
                if t1_counterpick_used:
                    if_win1_points1, if_win1_points2, if_win1_matchup = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    win1_points1 = 7.0 + if_win1_points1
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, True, t2_counterpick_used)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, True, t2_counterpick_used)
                    if 7.0 + keep_win_p1 > 6.0 + counterpick_p1:
                        if_win1_points1, if_win1_points2, if_win1_matchup = keep_win_p1, keep_win_p2, keep_win_m
                        win1_points1 = 7.0 + if_win1_points1
                    else:
                        if_win1_points1, if_win1_points2, if_win1_matchup = counterpick_p1, counterpick_p2, counterpick_m
                        win1_points1 = 6.0 + if_win1_points1
                
                # If team2 wins, consider counterpick option
                if t2_counterpick_used:
                    if_win2_points1, if_win2_points2, if_win2_matchup = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    win2_points2 = 7.0 + if_win2_points2
                else:
                    keep_win_p1, keep_win_p2, keep_win_m = calculate(new_t1, new_t2, 2, t1_original, t2_original, t1_counterpick_used, True)
                    counterpick_p1, counterpick_p2, counterpick_m = calculate(new_t1, new_t2, 1, t1_original, t2_original, t1_counterpick_used, True)
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
                    t1_idx = t1_original.index(player1)
                    t2_idx = t2_original.index(player2)
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
            
            # Create teams for remaining matches (add back banned players)
            t1_remaining = t1_for_match1[:i] + t1_for_match1[i+1:] + [t1[idx] for idx in t1_banned]
            t2_remaining = t2_for_match1[:j] + t2_for_match1[j+1:] + [t2[idx] for idx in t2_banned]
            
            # If team1 wins match 1, team1 must send first next
            # If team2 wins match 1, team2 must send first next
            if_win1_points1, if_win1_points2, if_win1_matchup = calculate(t1_remaining, t2_remaining, 1)
            if_win2_points1, if_win2_points2, if_win2_matchup = calculate(t1_remaining, t2_remaining, 2)
            
            # Expected value considering win probabilities
            total_points1 = match1_points1 + win_prob1 * if_win1_points1 + (1 - win_prob1) * if_win2_points1
            total_points2 = match1_points2 + win_prob1 * if_win1_points2 + (1 - win_prob1) * if_win2_points2
            
            # Find original indices for first match
            t1_idx = t1.index(player1) if player1 in t1 else -1
            t2_idx = t2.index(player2) if player2 in t2 else -1
            
            if total_points1 > best_total_points1:
                best_total_points1 = total_points1
                best_total_points2 = total_points2
                # Use the more likely outcome for matchup display
                if win_prob1 >= 0.5:
                    best_matchup = [(t1_idx, t2_idx)] + if_win1_matchup
                else:
                    best_matchup = [(t1_idx, t2_idx)] + if_win2_matchup
    
    return (best_total_points1, best_total_points2, best_matchup, t1_banned, t2_banned)


# Example usage
if __name__ == "__main__":
    # Example teams: each player is (rating, RD)
    team1 = [(2035, 200), (1955, 200), (1462, 200), (2060, 200), (2882, 200)]
    team2 = [(2387, 200), (2323, 200), (1880, 200), (2248, 200), (2086, 200)]
    
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
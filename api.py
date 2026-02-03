import requests
import time
from typing import List, Tuple
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

import os
import random
import time
from dotenv import load_dotenv

# Import from your updated engine.py
from engine import Player, Position, Team, Match, MatchAI, MatchPhase, EventType

def generate_argentina_roster():
    """Generates the exact 2026 World Cup 26-man roster for Argentina."""
    # Starting XI (Formation: 4-1-3-2)
    roster = [
        Player("Emiliano Martínez", Position.GOALKEEPER, 30, 90, 23),
        Player("Nahuel Molina", Position.DEFENDER, 60, 80, 26),
        Player("Cristian Romero", Position.DEFENDER, 55, 88, 13),
        Player("Lisandro Martínez", Position.DEFENDER, 65, 85, 6),
        Player("Nicolás Tagliafico", Position.DEFENDER, 60, 80, 3),
        Player("Leandro Paredes", Position.MIDFIELDER, 75, 82, 5),
        Player("Rodrigo De Paul", Position.MIDFIELDER, 80, 80, 7),
        Player("Enzo Fernández", Position.MIDFIELDER, 85, 75, 24),
        Player("Alexis Mac Allister", Position.MIDFIELDER, 82, 70, 20),
        Player("Lionel Messi", Position.FORWARD, 95, 30, 10),
        Player("Julián Alvarez", Position.FORWARD, 88, 40, 9),
        
        # Bench
        Player("Juan Musso", Position.GOALKEEPER, 20, 80, 1),
        Player("Gerónimo Rulli", Position.GOALKEEPER, 20, 78, 12),
        Player("Marcos Senesi", Position.DEFENDER, 50, 78, 2),
        Player("Gonzalo Montiel", Position.DEFENDER, 65, 77, 4),
        Player("Valentín Barco", Position.MIDFIELDER, 75, 70, 8),
        Player("Giovani Lo Celso", Position.MIDFIELDER, 80, 65, 11),
        Player("Exequiel Palacios", Position.MIDFIELDER, 78, 75, 14),
        Player("Nicolás González", Position.FORWARD, 82, 45, 15),
        Player("Thiago Almada", Position.FORWARD, 84, 40, 16),
        Player("Giuliano Simeone", Position.FORWARD, 78, 35, 17),
        Player("Nico Paz", Position.FORWARD, 75, 35, 18),
        Player("Nicolás Otamendi", Position.DEFENDER, 50, 85, 19),
        Player("José Manuel López", Position.FORWARD, 76, 30, 21),
        Player("Lautaro Martínez", Position.FORWARD, 89, 35, 22),
        Player("Facundo Medina", Position.DEFENDER, 50, 78, 25)
    ]
    return roster

def generate_egypt_roster():
    """Generates the exact 2026 World Cup 26-man roster for Egypt."""
    # Starting XI (Formation: 4-2-3-1)
    roster = [
        Player("Mostafa Shobeir", Position.GOALKEEPER, 25, 82, 23),
        Player("Mohamed Hany", Position.DEFENDER, 55, 75, 3),
        Player("Ramy Rabia", Position.DEFENDER, 45, 78, 5),
        Player("Yasser Ibrahim", Position.DEFENDER, 40, 80, 2),
        Player("Karim Hafez", Position.DEFENDER, 60, 74, 15),
        Player("Marwan Attia", Position.MIDFIELDER, 65, 78, 19),
        Player("Mohanad Lasheen", Position.MIDFIELDER, 68, 75, 17),
        Player("Emam Ashour", Position.MIDFIELDER, 75, 70, 8),
        Player("Mostafa Ziko", Position.MIDFIELDER, 72, 65, 11),
        Player("Haissem Hassan", Position.FORWARD, 78, 40, 12),
        Player("Mohamed Salah", Position.FORWARD, 92, 35, 10),
        
        # Bench
        Player("Mohamed El Shenawy", Position.GOALKEEPER, 20, 80, 1),
        Player("Hossam Abdelmaguid", Position.DEFENDER, 40, 76, 4),
        Player("Mohamed Abdelmonem", Position.DEFENDER, 50, 82, 6),
        Player("Trézéguet", Position.FORWARD, 80, 45, 7),
        Player("Hamza Abdelkarim", Position.FORWARD, 75, 30, 9),
        Player("Ahmed Fatouh", Position.DEFENDER, 65, 75, 13),
        Player("Hamdy Fathy", Position.MIDFIELDER, 70, 78, 14),
        Player("El Mahdy Soliman", Position.GOALKEEPER, 20, 75, 16),
        Player("Nabil Emad", Position.MIDFIELDER, 68, 74, 18),
        Player("Ibrahim Adel", Position.FORWARD, 77, 35, 20),
        Player("Mahmoud Saber", Position.MIDFIELDER, 70, 65, 21),
        Player("Omar Marmoush", Position.FORWARD, 82, 40, 22),
        Player("Tarek Alaa", Position.DEFENDER, 45, 72, 24),
        Player("Zizo", Position.FORWARD, 81, 45, 25),
        Player("Mohamed Alaa", Position.GOALKEEPER, 20, 72, 26)
    ]
    return roster

def main():
    print("Initializing FIFA World Cup Engine...")
    
    # 1. Load the API Key using dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Please check your .env file.")
        return

    # 2. Setup the Historic Teams
    argentina_roster = generate_argentina_roster()
    egypt_roster = generate_egypt_roster()
    
    # Since our Team class assigns the first 11 to the starting lineup,
    # the formations are naturally set based on how we built the lists above.
    home_team = Team("ARG", argentina_roster)
    away_team = Team("EGY", egypt_roster)

    # 3. Initialize the Match and AI Coaches
    match = Match(home_team, away_team)
    home_coach_ai = MatchAI(api_key=api_key, controlled_team=home_team)
    away_coach_ai = MatchAI(api_key=api_key, controlled_team=away_team)

    print(f"\n==================================================")
    print(f"🏆 KICKOFF! {home_team.country_name} vs {away_team.country_name}")
    print(f"🇦🇷 Argentina Formation: 4-1-3-2")
    print(f"🇪🇬 Egypt Formation: 4-2-3-1")
    print(f"==================================================")
    print("Simulating match... (This may take a minute due to AI API calls)")

    # 4. The Central State Engine Loop
    while match.phase == MatchPhase.REGULATION:
        match.run_minute_tick()

        home_team.risk_tolerance = home_coach_ai.risk_tolerance
        away_team.risk_tolerance = away_coach_ai.risk_tolerance
        
        if match.current_minute % 15 == 0:
            print(f"[{match.current_minute}'] AI Coaches are analyzing the pitch...")
            
            # Home Team AI Turn
            home_state = home_coach_ai.observe_state(match)
            home_action = home_coach_ai.decide_action(home_state)
            
            # Notice we pass match to apply_decision so it can log substitutions!
            home_coach_ai.apply_decision(home_action, match) 
            
            # Away Team AI Turn
            away_state = away_coach_ai.observe_state(match)
            away_action = away_coach_ai.decide_action(away_state)
            away_coach_ai.apply_decision(away_action, match)

    # 5. Handle Phase Transitions
    if match.home_score == match.away_score:
        match.resolve_regulation()

    # 6. Output the Definitive Match Timeline
    print("\n" + "="*50)
    print("🏆 FINAL MATCH TIMELINE 🏆")
    print("="*50)
    if not match.timeline:
        print("No major events occurred.")
    else:
        for event in match.timeline:
            print(event.to_string())

    # 7. Output the AI Decision Logs
    print("\n" + "="*50)
    print(f"🧠 {home_team.country_name} AI COACH LOG")
    print("="*50)
    for log in home_coach_ai.decision_log:
        print(log)

    print("\n" + "="*50)
    print(f"🧠 {away_team.country_name} AI COACH LOG")
    print("="*50)
    for log in away_coach_ai.decision_log:
        print(log)

if __name__ == "__main__":
    main()
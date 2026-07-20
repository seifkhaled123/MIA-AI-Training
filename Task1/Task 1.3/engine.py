from enum import Enum
import uuid
import random
from openai import OpenAI

from load_dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# --- ENUMS ---
class Position(Enum):
    FORWARD = "FORW ARD"
    MIDFIELDER = "MIDFIELDER"
    DEFENDER = "DEFENDER"
    GOALKEEPER = "GOALKEEPER"

class EventType(Enum):
    GOAL = "GOAL"
    SUBSTITUTION = "SUBSTITUTION"
    HALF_TIME = "HALF_TIME"
    FULL_TIME = "FULL_TIME"
    INCIDENT = "INCIDENT"

class MatchPhase(Enum):
    REGULATION = "REGULATION"
    FINISHED = "FINISHED"
    PENALTIES = "PENALTIES"

def print_penalty_board(home_team_name, away_team_name, round_results, current_score):
    """
    Reuses the tabular style from Task 1.1 to display penalty kick results.
    'O' means Goal, 'X' means Miss.
    """
    print("\n" + "="*35)
    print(" 🥅 PENALTY SHOOTOUT TRACKER 🥅 ")
    print("="*35)
    print(f"{'Round':<7} {home_team_name:<7} {away_team_name:<7} {'Score':<7}")
    
    for r in round_results:
        print(f"{r['round']:<7} {r['home_kick']:<7} {r['away_kick']:<7} {r['score']:<7}")
    print("="*35 + "\n")

# --- CORE CLASSES ---
class Player:
    def __init__(self, name: str, position: Position, base_attack: int, base_defense: int, jersey_number: int = None):
        self.name = name
        self.position = position
        
        # Fixed raw metrics scaled strictly from 1 to 100
        self.base_attack = max(1, min(100, base_attack)) 
        self.base_defense = max(1, min(100, base_defense))
        
        # Continuous dynamic parameter initialized to 100.0
        self.stamina = 100.0 

        # Initialize incidents for the Discipline System
        self.incidents = 0
        # Optional: Assign a jersey number for identification (1-99)
        self.jersey_number = jersey_number

    @property
    def display_name(self):
        """Helper to format the name with the jersey number."""
        if self.jersey_number is not None:
            return f"{self.name} (#{self.jersey_number})"
        return self.name
    
    def deplete_stamina(self, rate: float):
        """Deducts the designated rate, clamping the floor firmly at 10.0"""
        self.stamina = max(10.0, self.stamina - rate)

    def get_effective_attack(self):
        """Returns base_attack * (stamina / 100.0)"""
        return self.base_attack * (self.stamina / 100.0)

    def get_effective_defense(self):
        """Returns base_defense * (stamina / 100.0)"""
        return self.base_defense * (self.stamina / 100.0)

    def add_incident(self):
        """Adds an incident to the player. Returns the total incident count."""
        self.incidents += 1
        return self.incidents

class Team:
    def __init__(self, country_name: str, roster: list):
        self.country_name = country_name
        self.roster = roster # Full container of 26 Player objects
        
        # Initialize active lineup with exactly 11 players (e.g., the first 11)
        self.active_lineup = self.roster[:11] 
        self.substitutions_remaining = 5
        
    @property
    def bench(self):
        """Derived property: roster minus active_lineup"""
        return [player for player in self.roster if player not in self.active_lineup]

    def get_aggregate_attack(self):
        """Sums effective attack for FORWARD & MIDFIELDER, divided by their count."""
        attacking_players = [
            p for p in self.active_lineup 
            if p.position in (Position.FORWARD, Position.MIDFIELDER)
        ]
        count = len(attacking_players)
        
        # Prevent division by zero
        if count == 0:
            return 0.0
            
        total_attack = sum(p.get_effective_attack() for p in attacking_players)
        return total_attack / count

    def get_aggregate_defense(self):
        """Sums effective defense for DEFENDER & GOALKEEPER, divided by their count."""
        defending_players = [
            p for p in self.active_lineup 
            if p.position in (Position.DEFENDER, Position.GOALKEEPER)
        ]
        count = len(defending_players)
        
        if count == 0:
            return 0.0
            
        total_defense = sum(p.get_effective_defense() for p in defending_players)
        return total_defense / count

    def execute_substitution(self, player_out, player_in):
        """Swaps players if valid and decrements substitutions_remaining."""
        if self.substitutions_remaining <= 0:
            print(f"{self.country_name} has no substitutions remaining.")
            return False
            
        if player_out not in self.active_lineup:
            print(f"{player_out.display_name} is not on the pitch.")
            return False
            
        if player_in not in self.bench:
            print(f"{player_in.display_name} is not on the bench.")
            return False

        # Execute the swap
        index = self.active_lineup.index(player_out)
        self.active_lineup[index] = player_in
        self.substitutions_remaining -= 1
        return True

class MatchEvent:
    def __init__(self, event_type, minute: int, team, player, outcome_text: str):
        # Unique identifier assigned at creation; never reused
        self._event_id = str(uuid.uuid4()) 
        self._event_type = event_type
        self._minute = minute
        self._team = team
        # Player can be None for team-level events like HALF_TIME or FULL_TIME
        self._player = player 
        self._outcome_text = outcome_text

    # Exposing data as read-only properties to guarantee immutability
    @property
    def event_id(self): return self._event_id
    @property
    def event_type(self): return self._event_type
    @property
    def minute(self): return self._minute
    @property
    def team(self): return self._team
    @property
    def player(self): return self._player
    @property
    def outcome_text(self): return self._outcome_text

    def to_string(self):
        """The only method exposed. Formats the event into a standard string."""
        team_name = self._team.country_name if self._team else "System"
        player_name = self._player.display_name if self._player else "N/A"
        return f"[{self._minute}'] {self._event_type.value} - {team_name} ({player_name}): {self._outcome_text}"

class Match:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.current_minute = 0
        self.timeline = []
        self.phase = MatchPhase.REGULATION
        
        # Stamina points lost per minute 
        self.base_decay = 0.5 

    def run_minute_tick(self):
        # Stop ticking if the game is over
        if self.phase == MatchPhase.FINISHED:
            return

        self.current_minute += 1

        # 1. Stamina Degradation
        # Apply stamina decay to all active players on both teams
        for team in (self.home_team, self.away_team):
            for player in team.active_lineup:
                player.deplete_stamina(self.base_decay)

        # Discipline System Check
        self.process_discipline(self.home_team)
        self.process_discipline(self.away_team)

        # 2. Process goal attempts for both teams
        self.process_goal_attempt(self.home_team, self.away_team)
        self.process_goal_attempt(self.away_team, self.home_team)

        # 3. Phase Transition Workflow
        # Resolve the match outcome strictly at minute 90
        if self.current_minute == 90:
            self.phase = MatchPhase.FINISHED
            
            if self.home_score > self.away_score:
                result = f"{self.home_team.country_name} Wins!"
            elif self.away_score > self.home_score:
                result = f"{self.away_team.country_name} Wins!"
            else:
                result = "Match ends in a DRAW."
                
            # Log the Full Time event
            self.timeline.append(MatchEvent(
                event_type=EventType.FULL_TIME,
                minute=self.current_minute,
                team=None,
                player=None,
                outcome_text=f"FULL TIME: {result} (Score: {self.home_score}-{self.away_score})"
            ))

    def process_goal_attempt(self, attacking_team, defending_team):
        # 10% chance of generating a genuine scoring attempt each minute
        if random.random() < 0.10:
            agg_attack = attacking_team.get_aggregate_attack()
            agg_defense = defending_team.get_aggregate_defense()

            # Execute the Goal Event Determination formula
            attack_roll = agg_attack * random.uniform(0.75, 1.25)
            defense_roll = agg_defense * 1.3 * random.uniform(0.80, 1.20)

            if attack_roll > defense_roll:
                # Goal scored!
                if attacking_team == self.home_team:
                    self.home_score += 1
                else:
                    self.away_score += 1

                # Pick a random attacker to credit the goal to (for the timeline log)
                attackers = [p for p in attacking_team.active_lineup if p.position in (Position.FORWARD, Position.MIDFIELDER)]
                scorer = random.choice(attackers) if attackers else attacking_team.active_lineup[0]
                
                # Create and log the MatchEvent
                event = MatchEvent(
                    event_type=EventType.GOAL,
                    minute=self.current_minute,
                    team=attacking_team,
                    player=scorer,
                    outcome_text=f"GOAL! {scorer.display_name} blasts it past the keeper!"
                )
                self.timeline.append(event)

    def resolve_regulation(self):
        """Called at minute 90. Triggers penalties if tied."""
        if self.home_score != self.away_score:
            self.phase = MatchPhase.FINISHED
        else:
            self.phase = MatchPhase.PENALTIES
            self.run_penalty_shootout()

    def run_penalty_shootout(self):
        print(f"\n🚨 {self.home_team.country_name} and {self.away_team.country_name} are tied! Going to Penalties! 🚨")
        
        home_pens = 0
        away_pens = 0
        round_num = 1
        round_results = []
        
        # Standard Best of 5, then Sudden Death
        while True:
            # Simulate kicks (roughly 75% chance to score a penalty)
            home_goal = random.random() < 0.75
            away_goal = random.random() < 0.75
            
            if home_goal: home_pens += 1
            if away_goal: away_pens += 1
            
            # Log the round for the display board
            round_results.append({
                "round": round_num,
                "home_kick": "O" if home_goal else "X",
                "away_kick": "O" if away_goal else "X",
                "score": f"{home_pens}-{away_pens}"
            })
            
            # Print the live board using Task 1.1 style
            print_penalty_board(self.home_team.country_name, self.away_team.country_name, round_results, f"{home_pens}-{away_pens}")
            
            # Check for a winner after 5 rounds (or during sudden death)
            if round_num >= 5 and home_pens != away_pens:
                winner = self.home_team if home_pens > away_pens else self.away_team
                self.timeline.append(MatchEvent(
                    event_type=EventType.FULL_TIME,
                    minute=90, team=None, player=None,
                    outcome_text=f"{winner.country_name} wins on penalties ({home_pens}-{away_pens})!"
                ))
                self.phase = MatchPhase.FINISHED
                break
                
            round_num += 1

    def process_discipline(self, team):
        """2% chance per minute that a random active player commits a foul."""
        if random.random() < 0.02 and team.active_lineup:
            player = random.choice(team.active_lineup)
            incidents = player.add_incident()
            
            if incidents >= 2:
                # Remove the player from the pitch!
                team.active_lineup.remove(player)
                self.timeline.append(MatchEvent(
                    event_type=EventType.INCIDENT,
                    minute=self.current_minute,
                    team=team,
                    player=player,
                    outcome_text=f"RED CARD! {player.display_name} has committed a second foul and is sent off! {team.country_name} is down to {len(team.active_lineup)} men!"
                ))


class MatchAI:
    def __init__(self, api_key: str, controlled_team):
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.controlled_team = controlled_team
        
        # The 0.0-1.0 dial the model adjusts to play safer or more aggressively
        self.risk_tolerance = 0.5 
        
        # Human-readable record of tactical decisions
        self.decision_log = [] 

    def observe_state(self, match):
        """Serializes current score, minute, phase, and average stamina into a state vector."""
        # Prevent division by zero if the team got a bunch of red cards
        active_count = len(self.controlled_team.active_lineup)
        avg_stamina = sum(p.stamina for p in self.controlled_team.active_lineup) / active_count if active_count > 0 else 0
        
        # Figure out if the AI is winning or losing
        if self.controlled_team == match.home_team:
            score_status = "tied" if match.home_score == match.away_score else ("winning" if match.home_score > match.away_score else "losing")
        else:
            score_status = "tied" if match.home_score == match.away_score else ("winning" if match.away_score > match.home_score else "losing")
        
        state_prompt = (
            f"You are the manager of {self.controlled_team.country_name}. "
            f"Minute: {match.current_minute}/90. "
            f"Score: {match.home_score} - {match.away_score} (You are {score_status}). "
            f"Your team's average stamina is {avg_stamina:.1f}/100. "
            f"Your current risk tolerance is {self.risk_tolerance:.1f} (0.0 is safe, 1.0 is aggressive). "
            "TACTICAL GUIDELINES: "
            "- If average stamina drops below 70.0, output 'SUBSTITUTE' to bring on fresh legs. "
            "- If you are losing, output 'PUSH_ATTACK' to increase scoring chances. "
            "- If you are winning late in the game, output 'HOLD' to protect the lead. "
            "Based on this state, reply with EXACTLY ONE of the following commands and nothing else: "
            "SUBSTITUTE, CHANGE_FORMATION, HOLD, PUSH_ATTACK."
        )
        return state_prompt

    def decide_action(self, state_prompt):
        """Feeds the observed state into OpenAI and returns the decision."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # or "gpt-4o"
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a deterministic football manager AI. You must reply with exactly one of the allowed tactical commands and nothing else."
                    },
                    {
                        "role": "user", 
                        "content": state_prompt
                    }
                ],
                max_tokens=10,
                temperature=0.3 # Keep temperature low to prevent hallucinated formats
            )
            
            # Extract the response and clean it up
            action = response.choices[0].message.content.strip().upper()
            action = action.replace(".", "") # Strip periods just in case
            
            # Validate against the strictly permitted document commands
            if action in ["SUBSTITUTE", "CHANGE_FORMATION", "HOLD", "PUSH_ATTACK"]:
                return action
                
            return "HOLD" # Fallback if the AI hallucinates an invalid command
            
        except Exception as e:
            print(f"[MatchAI Error] API failed: {e}")
            return "HOLD" # Fallback on API failure/timeout

    def apply_decision(self, action, match):
        """Executes the chosen action against the controlled_team."""
        if action == "SUBSTITUTE":
            if self.controlled_team.substitutions_remaining > 0:
                # 1. Find the most tired player on the pitch
                lowest_stamina_active = min(self.controlled_team.active_lineup, key=lambda p: p.stamina)
                
                # 2. FIX: Find bench players with the EXACT SAME POSITION
                eligible_bench = [p for p in self.controlled_team.bench if p.position == lowest_stamina_active.position]
                
                if eligible_bench:
                    # Pick the highest-rated player from that specific position group
                    best_bench = max(eligible_bench, key=lambda p: p.base_attack + p.base_defense)
                    
                    # Execute the swap
                    self.controlled_team.execute_substitution(lowest_stamina_active, best_bench)
                    self.decision_log.append(f"[{action}] Subbed {lowest_stamina_active.display_name} OFF, {best_bench.display_name} ON.")
                    
                    # 3. FIX: Add the SUBSTITUTION event to the main match timeline!
                    match.timeline.append(MatchEvent(
                        event_type=EventType.SUBSTITUTION,
                        minute=match.current_minute,
                        team=self.controlled_team,
                        player=best_bench, # The specific player involved
                        outcome_text=f"TACTICAL SUB: {best_bench.display_name} comes on to replace {lowest_stamina_active.display_name}."
                    ))
                else:
                    self.decision_log.append(f"[SUBSTITUTE FAILED] No {lowest_stamina_active.position.value}s left on the bench.")
            else:
                self.decision_log.append("[SUBSTITUTE FAILED] No subs remaining.")

        elif action == "CHANGE_FORMATION":
            self.decision_log.append(f"[{action}] Shifted tactical formation.")

        elif action == "PUSH_ATTACK":
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.2)
            self.decision_log.append(f"[{action}] Increased risk tolerance to {self.risk_tolerance:.1f}.")

        elif action == "HOLD":
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.2)
            self.decision_log.append(f"[{action}] Decreased risk tolerance to {self.risk_tolerance:.1f}.")

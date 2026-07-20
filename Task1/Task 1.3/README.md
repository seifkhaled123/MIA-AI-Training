# 🏆 FIFA World Cup AI Simulation Engine

An advanced, object-oriented sports simulation engine featuring data-driven match calculations, tactical AI coaches powered by Large Language Models (LLMs), a complete discipline/injury system, and automated sudden-death penalty shootouts.

✨ **Latest Updates:** The system has been leveled up to include authentic 2026 World Cup 26-man rosters, precise jersey number tracking, and position-strict substitution logic!

---

## 🚀 Quick Start Guide

This project leverages **`uv`**, a fast Python package installer and resolver. Follow these steps to get the simulation up and running.

### 1. Prerequisites
* Ensure you have [uv Installed](https://github.com/astral-sh/uv).
* An OpenAI API Key.

### 2. Environment Setup
Create a `.env` file in the root directory of the project and insert your OpenAI API Key:

```env
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies
Use `uv` to sync your virtual environment using the existing `requirements.txt` file:

```bash
# Create a virtual environment
uv venv

# Install all required packages from requirements.txt
uv pip install -r requirements.txt
```

### 4. Run the Simulation
Execute the main orchestration script to kick off the historic Argentina vs. Egypt match simulation:

```bash
uv run main.py
```

---

## ⚙️ Core Engine Features

* **OOP Match Architecture:** Full object representations for `Player`, `Team`, `Match`, and `MatchEvent` with dynamic tracking of stamina decay.
* **Authentic Rosters:** Accurately mimics historic matchups (like the 2026 ARG vs. EGY game) with full 26-man rosters, starting XI formations, and real jersey numbers.
* **LLM Tactical Managers:** Real-time strategic evaluations every 15 minutes. AI coaches shift formations, balance risk tolerance, and make contextual substitutions based on game state.
* **Smart Substitution System:** Automatically replaces exhausted players with the highest-rated bench player of the *exact same position* while logging names and jersey numbers.
* **Discipline Simulation:** Realistic 2% per-minute foul calculation resulting in potential yellow/red cards and automatic field dismissal.
* **Sudden-Death Penalty Phase:** Authentic tournament tie-breaking logic with an interactive visual terminal dashboard tracking every single kick.

---

## 📊 Sample Output 

If you want to see the engine's capabilities without running the code or using an API key, here is a real terminal output showcasing a 0-0 tactical gridlock followed by a thrilling 8-round penalty shootout!

```text
uv run main.py
Initializing FIFA World Cup Engine...

==================================================
🏆 KICKOFF! ARG vs EGY
🇦🇷 Argentina Formation: 4-1-3-2
🇪🇬 Egypt Formation: 4-2-3-1
==================================================
Simulating match... (This may take a minute due to AI API calls)
[15'] AI Coaches are analyzing the pitch...
[30'] AI Coaches are analyzing the pitch...
[45'] AI Coaches are analyzing the pitch...
[60'] AI Coaches are analyzing the pitch...
[75'] AI Coaches are analyzing the pitch...
[90'] AI Coaches are analyzing the pitch...

🚨 ARG and EGY are tied! Going to Penalties! 🚨

===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
4       X       O       3-3    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
4       X       O       3-3    
5       X       X       3-3    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
4       X       O       3-3    
5       X       X       3-3    
6       O       O       4-4    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
4       X       O       3-3    
5       X       X       3-3    
6       O       O       4-4    
7       O       O       5-5    
===================================


===================================
 🥅 PENALTY SHOOTOUT TRACKER 🥅 
===================================
Round   ARG     EGY     Score  
1       O       O       1-1    
2       O       X       2-1    
3       O       O       3-2    
4       X       O       3-3    
5       X       X       3-3    
6       O       O       4-4    
7       O       O       5-5    
8       X       O       5-6    
===================================


==================================================
🏆 FINAL MATCH TIMELINE 🏆
==================================================
[60'] SUBSTITUTION - EGY (Mohamed El Shenawy (#1)): TACTICAL SUB: Mohamed El Shenawy (#1) comes on to replace Mostafa Shobeir (#23).
[75'] SUBSTITUTION - ARG (Juan Musso (#1)): TACTICAL SUB: Juan Musso (#1) comes on to replace Emiliano Martínez (#23).
[75'] SUBSTITUTION - EGY (Ahmed Fatouh (#13)): TACTICAL SUB: Ahmed Fatouh (#13) comes on to replace Mohamed Hany (#3).
[90'] FULL_TIME - System (N/A): FULL TIME: Match ends in a DRAW. (Score: 0-0)
[90'] SUBSTITUTION - ARG (Gonzalo Montiel (#4)): TACTICAL SUB: Gonzalo Montiel (#4) comes on to replace Nahuel Molina (#26).
[90'] SUBSTITUTION - EGY (Mohamed Abdelmonem (#6)): TACTICAL SUB: Mohamed Abdelmonem (#6) comes on to replace Ramy Rabia (#5).
[90'] FULL_TIME - System (N/A): EGY wins on penalties (5-6)!

==================================================
🧠 ARG AI COACH LOG
==================================================
[CHANGE_FORMATION] Shifted tactical formation.
[HOLD] Decreased risk tolerance to 0.3.
[HOLD] Decreased risk tolerance to 0.1.
[HOLD] Decreased risk tolerance to 0.0.
[SUBSTITUTE] Subbed Emiliano Martínez (#23) OFF, Juan Musso (#1) ON.
[SUBSTITUTE] Subbed Nahuel Molina (#26) OFF, Gonzalo Montiel (#4) ON.

==================================================
🧠 EGY AI COACH LOG
==================================================
[CHANGE_FORMATION] Shifted tactical formation.
[HOLD] Decreased risk tolerance to 0.3.
[HOLD] Decreased risk tolerance to 0.1.
[SUBSTITUTE] Subbed Mostafa Shobeir (#23) OFF, Mohamed El Shenawy (#1) ON.
[SUBSTITUTE] Subbed Mohamed Hany (#3) OFF, Ahmed Fatouh (#13) ON.
[SUBSTITUTE] Subbed Ramy Rabia (#5) OFF, Mohamed Abdelmonem (#6) ON.
```
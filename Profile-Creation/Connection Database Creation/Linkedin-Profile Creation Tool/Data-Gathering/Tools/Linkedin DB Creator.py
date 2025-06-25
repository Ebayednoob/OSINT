import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import time
import re
import threading
import asyncio

class PersonNode:
    """
    Represents a person in the network with a comprehensive, structured profile.
    """
    def __init__(self, profile_url, name="", username="", social_accounts=None, brief_description="", about="", 
                 services=None, activity_info="", experience=None, education=None, projects=None, skills=None, interests=None):
        self.profile_url = profile_url
        self.name = name
        self.username = username
        self.social_accounts = social_accounts if social_accounts else []
        self.brief_description = brief_description
        self.about = about
        self.services = services if services else []
        self.activity_info = activity_info
        self.experience = experience if experience else []
        self.education = education if education else []
        self.projects = projects if projects else []
        self.skills = skills if skills else []
        self.interests = interests if interests else []

    @classmethod
    def from_dict(cls, data):
        """Creates a PersonNode instance from a dictionary."""
        return cls(
            profile_url=data.get("profile_url", ""),
            name=data.get("name", "N/A"),
            username=data.get("username", ""),
            social_accounts=data.get("social_accounts", []),
            brief_description=data.get("brief_description", ""),
            about=data.get("about", ""),
            services=data.get("services", []),
            activity_info=data.get("activity_info", ""),
            experience=data.get("experience", []),
            education=data.get("education", []),
            projects=data.get("projects", []),
            skills=data.get("skills", []),
            interests=data.get("interests", [])
        )

    def to_dict(self):
        """Converts the PersonNode to a dictionary for serialization."""
        return {
            "profile_url": self.profile_url, "name": self.name, "username": self.username,
            "social_accounts": self.social_accounts, "brief_description": self.brief_description,
            "about": self.about, "services": self.services, "activity_info": self.activity_info,
            "experience": self.experience, "education": self.education, "projects": self.projects,
            "skills": self.skills, "interests": self.interests
        }

    def __str__(self):
        """String representation for display."""
        def format_list(title, items):
            if not items: return ""
            return f"\n\n{title}:\n" + ", ".join(items)
            
        def format_project_list(title, items):
            if not items: return ""
            return f"\n\n{title}:\n" + "\n---\n".join([f"{proj.get('name', 'N/A')}: {proj.get('description', '')}" for proj in items])

        exp_str = "\n\nExperience:\n" + "\n---\n".join([f"{exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('dates', 'N/A')})" for exp in self.experience])
        edu_str = "\n\nEducation:\n" + "\n---\n".join([f"{edu.get('school', 'N/A')}: {edu.get('degree', 'N/A')}" for edu in self.education])
        
        main_info = f"Name: {self.name}\nUsername: {self.username}\nDescription: {self.brief_description}\nActivity: {self.activity_info}\nURL: {self.profile_url}"
        socials = format_list("Social Accounts", self.social_accounts)
        about_sec = f"\n\nAbout:\n{self.about}"
        services_sec = format_list("Services", self.services)
        projects_sec = format_project_list("Projects", self.projects)
        skills_sec = format_list("Skills", self.skills)
        interests_sec = format_list("Interests", self.interests)

        return main_info + socials + about_sec + services_sec + exp_str + edu_str + projects_sec + skills_sec + interests_sec

class GraphExporter:
    """Generates a Cypher script for a PersonNode to be used in Neo4j."""
    def _escape(self, value):
        if value is None: return "null"
        return json.dumps(str(value))

    def generate_cypher(self, person):
        cypher = []
        props = f"p.name = {self._escape(person.name)}, p.username = {self._escape(person.username)}, p.brief_description = {self._escape(person.brief_description)}, p.about = {self._escape(person.about)}, p.activity_info = {self._escape(person.activity_info)}"
        cypher.append(f"// Profile for {person.name}")
        cypher.append(f"MERGE (p:Person {{profile_url: {self._escape(person.profile_url)}}}) ON CREATE SET {props} ON MATCH SET {props};")

        def create_relations(label, items, rel_type):
            if not items: return
            for item in items:
                cypher.append(f"MERGE (n:{label} {{name: {self._escape(item)}}});")
                cypher.append(f"MATCH (p:Person {{profile_url: {self._escape(person.profile_url)}}}), (n:{label} {{name: {self._escape(item)}}})")
                cypher.append(f"MERGE (p)-[:{rel_type}]->(n);")

        create_relations("Skill", person.skills, "HAS_SKILL")
        create_relations("Interest", person.interests, "IS_INTERESTED_IN")
        create_relations("Service", person.services, "PROVIDES_SERVICE")

        for exp in person.experience:
            if company := exp.get('company'):
                cypher.append(f"MERGE (c:Company {{name: {self._escape(company)}}});")
                cypher.append(f"MATCH (p:Person {{profile_url: {self._escape(person.profile_url)}}}), (c:Company {{name: {self._escape(company)}}})")
                cypher.append(f"MERGE (p)-[r:WORKS_AT]->(c) SET r.title = {self._escape(exp.get('title'))}, r.dates = {self._escape(exp.get('dates'))};")

        for edu in person.education:
            if school := edu.get('school'):
                cypher.append(f"MERGE (s:School {{name: {self._escape(school)}}});")
                cypher.append(f"MATCH (p:Person {{profile_url: {self._escape(person.profile_url)}}}), (s:School {{name: {self._escape(school)}}})")
                cypher.append(f"MERGE (p)-[r:STUDIED_AT]->(s) SET r.degree = {self._escape(edu.get('degree'))};")
        
        for proj in person.projects:
            if name := proj.get('name'):
                 cypher.append(f"MERGE (proj:Project {{name: {self._escape(name)}}});")
                 cypher.append(f"MATCH (p:Person {{profile_url: {self._escape(person.profile_url)}}}), (proj:Project {{name: {self._escape(name)}}})")
                 cypher.append(f"MERGE (p)-[r:WORKED_ON]->(proj) SET r.description = {self._escape(proj.get('description'))};")

        return "\n".join(cypher)

class LLMProcessor:
    """Uses the Gemini LLM to analyze unstructured text and extract structured data."""
    async def get_structured_data(self, raw_text, llm_settings):
        prompt = f"Analyze the following text and extract professional profile information according to the provided JSON schema. The text is likely a resume or professional bio. Populate all fields as accurately as possible.\n\nText to analyze:\n---\n{raw_text}\n---"
        return await self._call_gemini_with_schema(prompt, llm_settings)

    async def get_structured_data_from_edit(self, raw_text, llm_settings):
        prompt = f"The following text is an edited version of a user's profile. Please analyze it and re-structure it into the provided JSON schema. Ensure all sections (Experience, Education, etc.) are correctly parsed from the free-form text. The 'profile_url' field should be extracted from the 'URL:' line.\n\nEdited Text:\n---\n{raw_text}\n---"
        return await self._call_gemini_with_schema(prompt, llm_settings)


    async def _call_gemini_with_schema(self, prompt, llm_settings):
        chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "profile_url": {"type": "STRING"}, "name": {"type": "STRING"}, "username": {"type": "STRING"},
                "social_accounts": {"type": "ARRAY", "items": {"type": "STRING"}},
                "brief_description": {"type": "STRING"},
                "about": {"type": "STRING"},
                "services": {"type": "ARRAY", "items": {"type": "STRING"}},
                "activity_info": {"type": "STRING"},
                "experience": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"title": {"type": "STRING"}, "company": {"type": "STRING"}, "dates": {"type": "STRING"}}}},
                "education": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"school": {"type": "STRING"}, "degree": {"type": "STRING"}}}},
                "projects": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"name": {"type": "STRING"}, "description": {"type": "STRING"}}}},
                "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "interests": {"type": "ARRAY", "items": {"type": "STRING"}},
            }
        }

        payload = {"contents": chatHistory, "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}
        apiUrl = llm_settings.get("api_url")
        apiKey = llm_settings.get("api_key", "")
        
        try:
            import requests
            response = requests.post(f"{apiUrl}?key={apiKey}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
            else:
                return {"error": f"No valid response from LLM. Response: {result}"}
        except Exception as e:
            return {"error": f"API call failed: {e}"}

    async def chat_with_db(self, query, db_context, llm_settings):
        prompt = f"You are an expert AI assistant analyzing a database of professional profiles. Given the following database context in JSON format, please answer the user's question.\n\nDatabase Context:\n{db_context}\n\nUser Question: {query}"
        chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chatHistory}
        apiUrl = llm_settings.get("api_url")
        apiKey = llm_settings.get("api_key", "")

        try:
            import requests
            response = requests.post(f"{apiUrl}?key={apiKey}", headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Error: No valid response from LLM. Response: {result}"
        except Exception as e:
            return f"Error: API call failed: {e}"

class NetworkingAgent:
    """Manages the database of people and their associated files."""
    def __init__(self, logger=print):
        self.database = {}
        self.log = logger
        self.save_directory = os.path.join("Data-Gathering", "Data", "Linkedin")
        self.profile_data_directory = os.path.join(self.save_directory, "Profile Data")
        self.cypher_query_directory = os.path.join(self.save_directory, "Cypher Query")
        self.llm_processor = LLMProcessor()
        self.graph_exporter = GraphExporter()
        os.makedirs(self.profile_data_directory, exist_ok=True)
        os.makedirs(self.cypher_query_directory, exist_ok=True)

    def add_or_update_profile(self, person_node):
        if not person_node.profile_url: return False
        
        # If the name has changed, we might need to delete the old folder
        if (old_person := self.database.get(person_node.profile_url)) and old_person.name != person_node.name:
             self.delete_profile_folder(old_person)

        self.database[person_node.profile_url] = person_node
        self.log(f"Added/updated profile for {person_node.name}")
        self._save_individual_profile_data(person_node)
        self.save_main_database_index()
        return True

    def delete_profile_folder(self, person):
        """Deletes the data folder for a given person."""
        try:
            sanitized_name = re.sub(r'[\\/*?:"<>|]', "", person.name)
            profile_folder = os.path.join(self.profile_data_directory, sanitized_name)
            if os.path.exists(profile_folder):
                import shutil
                shutil.rmtree(profile_folder)
                self.log(f"Removed old data folder: {profile_folder}")
        except Exception as e:
            self.log(f"Error removing old profile folder: {e}")


    def _save_individual_profile_data(self, person):
        sanitized_name = re.sub(r'[\\/*?:"<>|]', "", person.name) if person.name else "Unnamed Profile"
        profile_folder = os.path.join(self.profile_data_directory, sanitized_name)
        os.makedirs(profile_folder, exist_ok=True)

        json_path = os.path.join(profile_folder, "profile.json")
        with open(json_path, 'w') as f: json.dump(person.to_dict(), f, indent=4)
        self.log(f"Saved JSON to {json_path}")

        cypher_script = self.graph_exporter.generate_cypher(person)
        cypher_path = os.path.join(profile_folder, f"{sanitized_name}.cypher")
        with open(cypher_path, 'w') as f: f.write(cypher_script)
        self.log(f"Saved Cypher graph query to {cypher_path}")
    
    def export_full_database_cypher(self):
        """Generates a single Cypher file for the entire database."""
        full_cypher = []
        for person in self.database.values():
            full_cypher.append(self.graph_exporter.generate_cypher(person))
        
        filepath = os.path.join(self.cypher_query_directory, "full_database.cypher")
        with open(filepath, 'w') as f:
            f.write("\n\n".join(full_cypher))
        self.log(f"Full database Cypher script exported to {filepath}")
        return filepath


    def save_main_database_index(self, filename="linkedin_database_index.json"):
        os.makedirs(self.save_directory, exist_ok=True)
        filepath = os.path.join(self.save_directory, filename)
        with open(filepath, "w") as f: json.dump({url: person.name for url, person in self.database.items()}, f, indent=4)
        self.log(f"Saved main database index to {filepath}")

    def load_database(self, filename="linkedin_database_index.json"):
        filepath = os.path.join(self.save_directory, filename)
        try:
            with open(filepath, "r") as f: index_data = json.load(f)
            for url, name in index_data.items():
                sanitized_name = re.sub(r'[\\/*?:"<>|]', "", name)
                json_path = os.path.join(self.profile_data_directory, sanitized_name, "profile.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as pf:
                        person_data = json.load(pf)
                        self.database[url] = PersonNode.from_dict(person_data)
                else:
                    self.log(f"Warning: Profile JSON not found for {name} at {json_path}")
            self.log(f"Database loaded from index. Found {len(self.database)} entries.")
        except FileNotFoundError: self.log(f"No database index found at {filepath}.")
        except Exception as e: self.log(f"Error loading database: {e}")

class ManualProfileWindow(tk.Toplevel):
    def __init__(self, master, agent, prefill_data=None):
        super().__init__(master)
        self.title("Manual Profile Entry")
        self.geometry("800x800")
        self.agent = agent
        self.master_app = master
        self.create_widgets()
        if prefill_data: self.populate_form(prefill_data)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- Create Tabs for Entry ---
        tab1 = ttk.Frame(notebook, padding="10")
        tab2 = ttk.Frame(notebook, padding="10")
        tab3 = ttk.Frame(notebook, padding="10")
        notebook.add(tab1, text="Primary Info")
        notebook.add(tab2, text="Professional Details")
        notebook.add(tab3, text="Skills & Interests")
        
        self._create_primary_tab(tab1)
        self._create_professional_tab(tab2)
        self._create_skills_tab(tab3)

        ttk.Button(main_frame, text="Save Profile", command=self._save_profile).pack(side=tk.RIGHT, pady=10)

    def _create_primary_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        fields = ["Name", "Username", "Brief Description", "Activity Info", "Profile URL"]
        self.entries = {}
        for i, field in enumerate(fields):
            ttk.Label(parent, text=f"{field}:").grid(row=i, column=0, sticky="nw", pady=2, padx=5)
            entry = ttk.Entry(parent)
            entry.grid(row=i, column=1, sticky="ew", pady=2)
            self.entries[field] = entry
        
        ttk.Label(parent, text="Social Accounts (one per line):").grid(row=len(fields), column=0, sticky="nw", pady=5, padx=5)
        self.social_text = scrolledtext.ScrolledText(parent, height=4, wrap=tk.WORD)
        self.social_text.grid(row=len(fields), column=1, sticky="ew", pady=2)
        
        ttk.Label(parent, text="About:").grid(row=len(fields)+1, column=0, sticky="nw", pady=5, padx=5)
        self.about_text = scrolledtext.ScrolledText(parent, height=10, wrap=tk.WORD)
        self.about_text.grid(row=len(fields)+1, column=1, sticky="ew", pady=2)
        
    def _create_professional_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        exp_label = "Experience (Format: title;company;dates per line):"
        ttk.Label(parent, text=exp_label).pack(anchor="w")
        self.experience_text = scrolledtext.ScrolledText(parent, height=10, wrap=tk.WORD)
        self.experience_text.pack(fill=tk.X, expand=True, pady=(0, 10))

        edu_label = "Education (Format: school;degree per line):"
        ttk.Label(parent, text=edu_label).pack(anchor="w")
        self.education_text = scrolledtext.ScrolledText(parent, height=5, wrap=tk.WORD)
        self.education_text.pack(fill=tk.X, expand=True, pady=(0, 10))
        
        proj_label = "Projects (Format: name;description per line):"
        ttk.Label(parent, text=proj_label).pack(anchor="w")
        self.projects_text = scrolledtext.ScrolledText(parent, height=7, wrap=tk.WORD)
        self.projects_text.pack(fill=tk.X, expand=True)

    def _create_skills_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="Services (one per line):").pack(anchor="w")
        self.services_text = scrolledtext.ScrolledText(parent, height=6, wrap=tk.WORD)
        self.services_text.pack(fill=tk.X, expand=True, pady=(0, 10))

        ttk.Label(parent, text="Skills (comma-separated):").pack(anchor="w")
        self.skills_text = scrolledtext.ScrolledText(parent, height=6, wrap=tk.WORD)
        self.skills_text.pack(fill=tk.X, expand=True, pady=(0, 10))

        ttk.Label(parent, text="Interests (comma-separated):").pack(anchor="w")
        self.interests_text = scrolledtext.ScrolledText(parent, height=6, wrap=tk.WORD)
        self.interests_text.pack(fill=tk.X, expand=True, pady=(0, 10))

    def populate_form(self, data):
        self.entries["Name"].insert(0, data.get("name", ""))
        self.entries["Username"].insert(0, data.get("username", ""))
        self.entries["Brief Description"].insert(0, data.get("brief_description", ""))
        self.entries["Activity Info"].insert(0, data.get("activity_info", ""))
        self.entries["Profile URL"].insert(0, data.get("profile_url", ""))
        self.social_text.insert("1.0", "\n".join(data.get("social_accounts", [])))
        self.about_text.insert("1.0", data.get("about", ""))
        
        self.experience_text.insert("1.0", "\n".join([f"{e.get('title','')};{e.get('company','')};{e.get('dates','')}" for e in data.get("experience", [])]))
        self.education_text.insert("1.0", "\n".join([f"{e.get('school','')};{e.get('degree','')}" for e in data.get("education", [])]))
        self.projects_text.insert("1.0", "\n".join([f"{p.get('name','')};{p.get('description','')}" for p in data.get("projects", [])]))
        
        self.services_text.insert("1.0", "\n".join(data.get("services", [])))
        self.skills_text.insert("1.0", ", ".join(data.get("skills", [])))
        self.interests_text.insert("1.0", ", ".join(data.get("interests", [])))

    def _parse_multiline_text(self, text, keys):
        items = []
        for line in text.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split(';')
            items.append({keys[i]: parts[i].strip() if i < len(parts) else '' for i in range(len(keys))})
        return items

    def _save_profile(self):
        url = self.entries["Profile URL"].get()
        if not url: messagebox.showerror("Input Error", "Profile URL is required.", parent=self); return
        new_node = PersonNode(
            profile_url=url, name=self.entries["Name"].get(), username=self.entries["Username"].get(),
            brief_description=self.entries["Brief Description"].get(),
            activity_info=self.entries["Activity Info"].get(),
            social_accounts=[s.strip() for s in self.social_text.get("1.0", tk.END).strip().split('\n') if s.strip()],
            about=self.about_text.get("1.0", tk.END).strip(),
            experience=self._parse_multiline_text(self.experience_text.get("1.0", tk.END), ['title', 'company', 'dates']),
            education=self._parse_multiline_text(self.education_text.get("1.0", tk.END), ['school', 'degree']),
            projects=self._parse_multiline_text(self.projects_text.get("1.0", tk.END), ['name', 'description']),
            services=[s.strip() for s in self.services_text.get("1.0", tk.END).strip().split('\n') if s.strip()],
            skills=[s.strip() for s in self.skills_text.get("1.0", tk.END).strip().split(',') if s.strip()],
            interests=[i.strip() for i in self.interests_text.get("1.0", tk.END).strip().split(',') if i.strip()]
        )
        if self.agent.add_or_update_profile(new_node):
            self.master_app.update_database_list()
            self.master_app.select_person_in_list(new_node)
            self.destroy()

class BulkDataWindow(tk.Toplevel):
    def __init__(self, master, agent, llm_settings):
        super().__init__(master)
        self.title("Process Bulk Data with AI")
        self.geometry("700x500")
        self.agent = agent
        self.master_app = master
        self.llm_settings = llm_settings
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="Paste raw text (resume, bio, etc.) below:").pack(anchor="w")
        self.text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.process_button = ttk.Button(main_frame, text="Process with Gemini", command=self.process_text)
        self.process_button.pack(anchor="e")

    def process_text(self):
        raw_text = self.text_area.get("1.0", tk.END)
        if not raw_text.strip(): messagebox.showwarning("Input Error", "Please paste some text to process.", parent=self); return
        self.process_button.config(state="disabled")
        self.master_app.log("Sending text to Gemini for processing...")
        def task():
            llm_data = asyncio.run(self.agent.llm_processor.get_structured_data(raw_text, self.llm_settings))
            self.after(0, self.on_processing_complete, llm_data)
        threading.Thread(target=task).start()

    def on_processing_complete(self, data):
        self.process_button.config(state="normal")
        if "error" in data:
            self.master_app.log(f"LLM Error: {data['error']}")
            messagebox.showerror("AI Processing Failed", data['error'], parent=self)
        else:
            self.master_app.log("LLM processing complete. Opening manual entry form with pre-filled data.")
            ManualProfileWindow(self.master_app, self.agent, prefill_data=data)
            self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Comprehensive Networking Database")
        self.geometry("1100x800")
        self.agent = NetworkingAgent(logger=self.log)
        self.create_widgets()
        self.agent.load_database()
        self.update_database_list()
        self.currently_selected_url = None

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create Tabs
        self.chat_tab = ttk.Frame(notebook)
        self.data_proc_tab = ttk.Frame(notebook)
        self.cypher_tab = ttk.Frame(notebook)
        self.llm_setup_tab = ttk.Frame(notebook)
        notebook.add(self.chat_tab, text="Database Chat")
        notebook.add(self.data_proc_tab, text="Data Processing")
        notebook.add(self.cypher_tab, text="Cypher Exporter")
        notebook.add(self.llm_setup_tab, text="LLM Setup")

        # --- Populate Tabs ---
        self._create_chat_tab()
        self._create_data_proc_tab()
        self._create_cypher_tab()
        self._create_llm_setup_tab()

        # --- Log Widget ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.X, expand=True)

    def _create_chat_tab(self):
        chat_frame = ttk.Frame(self.chat_tab, padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state="disabled", height=15)
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=5)
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=5)
        self.chat_input = ttk.Entry(input_frame)
        self.chat_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_chat_message)
        self.send_button.pack(side=tk.RIGHT)

    def _create_data_proc_tab(self):
        proc_frame = ttk.Frame(self.data_proc_tab, padding="10")
        proc_frame.pack(fill=tk.BOTH, expand=True)
        
        action_frame = ttk.Frame(proc_frame, padding="5")
        action_frame.pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="Create New Profile", command=self.open_manual_entry).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Process Bulk Data with AI", command=self.open_bulk_processor).pack(side=tk.LEFT, padx=5)
        self.edit_button = ttk.Button(action_frame, text="Edit Selected Profile", command=self.toggle_edit_mode)
        self.edit_button.pack(side=tk.LEFT, padx=5)
        self.process_changes_button = ttk.Button(action_frame, text="Process Changes with AI", command=self.process_edited_profile, state="disabled")
        self.process_changes_button.pack(side=tk.LEFT, padx=5)

        content_frame = ttk.Frame(proc_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        content_frame.grid_columnconfigure(0, weight=1); content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)
        
        db_frame = ttk.LabelFrame(content_frame, text="Database Profiles", padding="10")
        db_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.db_listbox = tk.Listbox(db_frame)
        self.db_listbox.pack(fill=tk.BOTH, expand=True)
        self.db_listbox.bind('<<ListboxSelect>>', self.on_select_person)
        
        profile_frame = ttk.LabelFrame(content_frame, text="Profile Details", padding="10")
        profile_frame.grid(row=0, column=1, sticky="nsew")
        self.profile_details_text = scrolledtext.ScrolledText(profile_frame, wrap=tk.WORD, state='disabled')
        self.profile_details_text.pack(fill=tk.BOTH, expand=True)

    def _create_cypher_tab(self):
        cypher_frame = ttk.Frame(self.cypher_tab, padding="10")
        cypher_frame.pack(fill=tk.BOTH, expand=True)
        
        cypher_actions = ttk.Frame(cypher_frame, padding="5")
        cypher_actions.pack(fill=tk.X, pady=5)
        ttk.Button(cypher_actions, text="Generate for Selected Profile", command=self.generate_cypher_for_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(cypher_actions, text="Export Full Database to Cypher", command=self.export_full_cypher).pack(side=tk.LEFT, padx=5)

        self.cypher_text = scrolledtext.ScrolledText(cypher_frame, wrap=tk.WORD, state='disabled')
        self.cypher_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def _create_llm_setup_tab(self):
        setup_frame = ttk.Frame(self.llm_setup_tab, padding="20")
        setup_frame.pack(fill=tk.BOTH, expand=True)
        setup_frame.columnconfigure(1, weight=1)
        
        ttk.Label(setup_frame, text="Model Name:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        self.model_name_entry = ttk.Entry(setup_frame, width=80)
        self.model_name_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self.model_name_entry.insert(0, "gemini-2.0-flash")

        ttk.Label(setup_frame, text="Custom API URL (optional):").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        self.custom_api_url_entry = ttk.Entry(setup_frame, width=80)
        self.custom_api_url_entry.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(setup_frame, text="API Key (optional):").grid(row=2, column=0, sticky="w", pady=5, padx=5)
        self.api_key_entry = ttk.Entry(setup_frame, width=80, show="*")
        self.api_key_entry.grid(row=2, column=1, sticky="ew", pady=5)
        
        ttk.Button(setup_frame, text="Save Settings", command=self.save_llm_settings).grid(row=3, column=1, sticky="e", pady=10)
        ttk.Label(setup_frame, text="Note: For standard Google models, only the Model Name is needed.\nUse the Custom API URL for other endpoints.").grid(row=4, column=0, columnspan=2, pady=20, sticky="w")
    
    def get_llm_settings(self):
        custom_url = self.custom_api_url_entry.get().strip()
        model_name = self.model_name_entry.get().strip()
        
        if custom_url:
            api_url = custom_url
        else:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        return {"api_url": api_url, "api_key": self.api_key_entry.get()}

    def save_llm_settings(self):
        self.log("LLM settings have been updated in the UI. They will be used on the next request.")
        messagebox.showinfo("Settings Updated", "LLM settings will be used for the next AI request.")

    def log(self, message):
        if hasattr(self, 'log_text'):
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
            self.update_idletasks()

    def send_chat_message(self):
        query = self.chat_input.get()
        if not query.strip(): return
        self.chat_input.delete(0, tk.END)
        self.send_button.config(state="disabled")

        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, f"You: {query}\n\n", "user_message")
        self.chat_history.config(state="disabled")
        
        db_context = json.dumps([p.to_dict() for p in self.agent.database.values()], indent=2)

        def task():
            response = asyncio.run(self.agent.llm_processor.chat_with_db(query, db_context, self.get_llm_settings()))
            self.after(0, self.on_chat_response, response)
        
        threading.Thread(target=task).start()

    def on_chat_response(self, response):
        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, f"AI: {response}\n\n", "ai_message")
        self.chat_history.config(state="disabled")
        self.send_button.config(state="normal")

    def open_manual_entry(self): ManualProfileWindow(self, self.agent)
    def open_bulk_processor(self): BulkDataWindow(self, self.agent, self.get_llm_settings())
    
    def toggle_edit_mode(self):
        """Toggles the state of the profile details text widget."""
        if not self.currently_selected_url:
            messagebox.showinfo("Action Required", "Please select a profile to edit.")
            return

        current_state = self.profile_details_text.cget("state")
        new_state = "normal" if current_state == "disabled" else "disabled"
        self.profile_details_text.config(state=new_state)
        
        process_button_state = "normal" if new_state == "normal" else "disabled"
        self.process_changes_button.config(state=process_button_state)
        self.log(f"Profile view set to {'EDIT' if new_state == 'normal' else 'READ-ONLY'} mode.")

    def process_edited_profile(self):
        """Sends the edited text to the LLM for structuring and updates the profile."""
        edited_text = self.profile_details_text.get("1.0", tk.END)
        if not edited_text.strip(): messagebox.showerror("Error", "No text to process."); return
        if not self.currently_selected_url: messagebox.showerror("Error", "No profile selected for edit."); return

        self.process_changes_button.config(state="disabled")
        self.log("Sending edited profile to Gemini for structuring...")

        original_url_to_update = self.currently_selected_url

        def task():
            llm_data = asyncio.run(self.agent.llm_processor.get_structured_data_from_edit(edited_text, self.get_llm_settings()))
            self.after(0, self.on_edit_processing_complete, llm_data, original_url_to_update)

        threading.Thread(target=task).start()

    def on_edit_processing_complete(self, llm_data, original_url):
        if "error" in llm_data:
            self.log(f"LLM Error during edit processing: {llm_data['error']}")
            messagebox.showerror("AI Processing Failed", llm_data['error'])
            self.process_changes_button.config(state="normal") # Re-enable on failure
            return

        new_url = llm_data.get("profile_url")
        if new_url and new_url != original_url:
            if new_url in self.agent.database:
                messagebox.showerror("Update Failed", f"A profile with the new URL '{new_url}' already exists.")
                self.process_changes_button.config(state="normal")
                return
            del self.agent.database[original_url]
            self.log(f"Profile URL changed. Old entry for {original_url} removed.")
        
        updated_node = PersonNode.from_dict(llm_data)
        self.agent.add_or_update_profile(updated_node)
        
        self.log("Profile successfully updated by AI.")
        self.update_database_list()
        self.select_person_in_list(updated_node)
        self.toggle_edit_mode() # Return to read-only mode


    def generate_cypher_for_selected(self):
        if not (selected_url := self.currently_selected_url):
            messagebox.showinfo("Action Required", "Please select a profile from the list first.")
            return
        person = self.agent.database[selected_url]
        cypher = self.agent.graph_exporter.generate_cypher(person)
        
        self.cypher_text.config(state="normal")
        self.cypher_text.delete("1.0", tk.END)
        self.cypher_text.insert("1.0", cypher)
        self.cypher_text.config(state="disabled")
        self.log(f"Generated Cypher query for {person.name}.")

    def export_full_cypher(self):
        if not self.agent.database:
            messagebox.showinfo("Database Empty", "There are no profiles in the database to export.")
            return
        filepath = self.agent.export_full_database_cypher()
        messagebox.showinfo("Export Complete", f"Full database Cypher script saved to:\n{filepath}")

    def get_sorted_db_keys(self): return sorted(self.agent.database.keys())

    def update_database_list(self):
        self.db_listbox.delete(0, tk.END)
        sorted_keys = self.get_sorted_db_keys()
        for url in sorted_keys:
            person = self.agent.database.get(url)
            if isinstance(person, PersonNode): # Defensive check
                self.db_listbox.insert(tk.END, f"{person.name} - {person.brief_description[:50]}...")
            else:
                self.log(f"Warning: Corrupt data in database for URL {url}. Found type {type(person)}.")

    def select_person_in_list(self, person_node):
        sorted_keys = self.get_sorted_db_keys()
        if person_node.profile_url in sorted_keys:
            idx = sorted_keys.index(person_node.profile_url)
            self.db_listbox.selection_clear(0, tk.END)
            self.db_listbox.selection_set(idx)
            self.on_select_person(None) # Trigger display update

    def on_select_person(self, event):
        if self.profile_details_text.cget("state") == "normal":
             self.toggle_edit_mode() # Exit edit mode if a new person is selected

        if not (selection_indices := self.db_listbox.curselection()):
            self.currently_selected_url = None
            return

        selected_url = self.get_sorted_db_keys()[selection_indices[0]]
        self.currently_selected_url = selected_url
        person = self.agent.database[selected_url]
        self.display_person_details(person)
        if self.cypher_text.winfo_viewable():
            self.generate_cypher_for_selected()

    def display_person_details(self, person):
        self.profile_details_text.configure(state='normal')
        self.profile_details_text.delete(1.0, tk.END)
        self.profile_details_text.insert(tk.END, str(person))
        self.profile_details_text.configure(state='disabled')

if __name__ == '__main__':
    app = App()
    app.mainloop()

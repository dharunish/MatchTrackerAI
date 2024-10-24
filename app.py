###################################################################
# Import required libraries                                       #
###################################################################
import gradio as gr
from video_analysis import *
import json
from json.decoder import JSONDecodeError
import matplotlib.pyplot as plt
from collections import defaultdict


###################################################################
# Initialize variables for keeping track of players and positions #
###################################################################
ps = ['Pick', 'GK','LB','CB','RB','RM','CM','LM','CAM','LW','RW','CF','ST'] #Positions
player_list = [] # list to store players

player_list.append(  { 'name': 'Lionel Messi' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name': 'Cristiano Ronaldo' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Erling Haaland' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Jude Bellingham' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Kevin De Bruyne' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Lamine Yamal' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Virgil Van Dijk' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Marc Ter Stegen' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Mohamed Salah' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Kylian Mbappe' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
player_list.append(  { 'name' : 'Jamal Musiala' ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )


player_list.append(  { 'name' : 'Neymar Jr.' ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )
player_list.append(  { 'name' : 'Ousmane Dembele' ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )
player_list.append(  { 'name' : 'Christian Pulisic' ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )
player_list.append(  { 'name' : 'Antony' ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )
player_list.append(  { 'name' : 'Harry Maguire' ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )

message = '' #to store the substitution history


###################################################################
# Function to plot the play time statistics                       #
###################################################################
def plot_play_time(data,game):
    if game == "All Games":  # Aggregate playtime across all games
        player_playtime = defaultdict(int)
        for players in data.values():
            for player in players:
                player_playtime[player['name']] += player['playtime']
        # Prepare data for plotting
        names = list(player_playtime.keys())
        playtimes = list(player_playtime.values())
    else:  # If a specific game is selected
        players = data.get(game, [])
        names = [player['name'] for player in players]
        playtimes = [player['playtime'] for player in players]


    plt.figure(figsize=(10, 8)) 
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=60,ha='right')
    colors = plt.cm.viridis(np.linspace(0, 1, len(players)))
    plt.bar(names, playtimes,  color=colors)
    plt.title(f'Playtime for {game}')
    plt.xlabel('Players')
    plt.ylabel('Playtime')
    plt.ylim(0, max(playtimes) + 5)
    plt.grid(axis='y')
    return plt


###################################################################
# Main gradio UI logic                                            #
###################################################################
with gr.Blocks() as app:
    with gr.Row():
        gr.Image("Models/Title.png", container=False)     
    with gr.Tab('Team Info'):
        with gr.Row():
            txtname = []
            with gr.Column():
                gr.Markdown('Starting XI')
                for i in range(11):
                    with gr.Row():
                        txtname.append(gr.Textbox(interactive=True, container=False, value = player_list[i]['name'], placeholder='Enter Player Name'))
            with gr.Column():
                gr.Markdown('Bench')
                for i in range(5):
                    with gr.Row():
                        txtname.append(gr.Textbox(interactive=True, container=False, value = player_list[i+11]['name'],placeholder='Enter Player Name'))
        with gr.Row():
            def save_info(*txtname):
                player_list = []
                for i in range(11):
                    player_list.append(  { 'name':txtname[i] ,  'position':None, 'playtime':0, 'playing': True   ,  'selected': False} )
                for i in range(5):
                    player_list.append(  { 'name':txtname[11+i] ,  'position':None, 'playtime':0, 'playing': False   ,  'selected': False} )
                return player_list
            player_list = gr.State(player_list)
            gr.Button(value='Save Info', variant='primary').click(save_info, inputs=txtname, outputs=[player_list])
    with gr.Tab('Substitution'):

        game = gr.State('not started')
        with gr.Row():
            gInfo = gr.Textbox(placeholder="Enter Game Info", interactive=True, container=False)
            gr.Button('Kickoff', variant='primary').click(lambda game: 'running', inputs=[game], outputs=[game])
            gr.Button('Halftime', variant='secondary').click(lambda game: 'paused', inputs=[game], outputs=[game])
            def finish(game, gInfo, players):
                game = 'finished'
                file = {}
                with open("results.json") as outfile: 
                    try:
                        file = json.load(outfile)
                    except JSONDecodeError:
                        pass
                with open("results.json", "w") as outfile:
                    file[gInfo] = players
                    json.dump(file, outfile, indent=4)
                return game, "Game stats saved !"
            gr.Button('Finish', variant='stop').click(finish, inputs=[game, gInfo, player_list], outputs=[game, gInfo])
        def timer(game, players):
            if game == 'running':
                for p in players:
                    if p['playing']:
                        p['playtime'] += 2
            return players
        gr.Timer(2).tick(timer, inputs=[game, player_list], outputs=[player_list])
        @gr.render(inputs=player_list)
        def updateScreen(players):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("# On Field", container=False)
                    with gr.Row():
                        gr.Markdown('Player Name', container=False)
                        gr.Markdown('Play Time', container=False)
                        gr.Markdown('    Position', container=False)
                    for i,p in enumerate(players):
                        if p['playing']:
                            with gr.Row():
                                cb = gr.Checkbox(label = p['name'], scale=0, value=p['selected'] )
                                def selected(cb, p=p):
                                    if cb:
                                        p['selected'] = True
                                    else:
                                        p['selected'] = False
                                    return players
                                cb.select(selected, inputs=[cb], outputs = [player_list])
                                gr.Textbox(p['playtime'], scale=0,container=False)
                            
                                dd = gr.Dropdown(value = p['position'], choices = ps, scale=0,  container=False)
                                def dd_change(dd,p=p):
                                    global message
                                    p['position'] = dd
                                    message += f"{p['name']} moved to position {p['position']}\n"
                                    return players
                                dd.change(dd_change, inputs=[dd], outputs=[player_list])
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("# On Bench", container=False)
                    with gr.Row():
                        gr.Markdown('Player Name', container=False)
                        gr.Markdown(' Play Time', container=False)
                    for i,p in enumerate(players):
                        if not p['playing']:
                            with gr.Row():
                                cb = gr.Checkbox(label = p['name'], scale=0, value=p['selected'])
                                def selected(cb, p=p):
                                    if cb:
                                        p['selected'] = True
                                    else:
                                        p['selected'] = False
                                    return players
                                cb.select(selected, inputs=[cb], outputs = [player_list])
                                gr.Textbox(value = p['playtime'], scale=0, container=False)
            
            with gr.Row():
                def sub(players):
                    global message
                    n = ''
                    currPos = ''
                    o = ''
                    pNum = 0
                    for i, p in enumerate(players):
                        if p['selected']:
                            if p['playing'] == True:
                                p['playing'] = False
                                o = f'{p['name']} at {p['position']}'
                                currPos = p['position']
                            else:
                                p['playing'] = True
                                n = p['name']
                                pNum = i
                                
                            p['selected'] = False
                            players[pNum]['position'] = currPos

                    message += f'{n} subs in for {o}\n'
                    return players
                gr.Button(value="Sub", variant='primary').click(sub, inputs=[player_list], outputs=[player_list])

            with gr.Row():
                gr.Markdown('# Substitution History')
            with gr.Row():
                gr.Textbox(value=message, interactive=True, container=False)      
    with gr.Tab('Video Analysis'):
        with gr.Row():
            with gr.Column():
                infile = gr.Video(label='Input Video', show_label=True, height=600, width=700, container=True)
                with gr.Row():
                    process_button = gr.Button("Analyze Video")
                    stop_button = gr.Button('Cancel')
            with gr.Column():
                finalout = gr.Video(value='OutputVideos/output_video.mp4', label='Output Video', show_label=True, height=600, width=700,visible=True, container=True)
                process_action = process_button.click(fn=run_projection,inputs=[infile],outputs=[finalout])
                stop_button.click(fn=None, inputs=None, outputs=None, cancels=[process_action])    
    with gr.Tab('Statistics'):
        gr.Markdown("# Player Play Time Visualization")
        with open('results.json', 'r') as file:
            data = json.load(file)
        game_options = list(data.keys()) + ["All Games"]

        selected_game = gr.Dropdown(choices=game_options, value=game_options[0], label="Select Game")
        plot_button = gr.Button("Show Play Time", variant='primary')
        plot_output = gr.Plot(label="Play Time Statistics")

        def update_plot(selected_game, data=data):
            return plot_play_time(data,selected_game)

        plot_button.click(update_plot, inputs=selected_game, outputs=plot_output)

###################################################################
# Lauch the app                                                   #
###################################################################
app.launch(share=True)
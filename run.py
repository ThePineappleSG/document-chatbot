﻿yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"
red = "\033[0;31m"

text =f"""{red}
████████╗██╗  ██╗███████╗     █████╗ ██████╗ ██╗   ██╗███████╗███╗   ██╗████████╗██╗   ██╗███████╗        
╚══██╔══╝██║  ██║██╔════╝    ██╔══██╗██╔══██╗██║   ██║██╔════╝████╗  ██║╚══██╔══╝██║   ██║██╔════╝        
   ██║   ███████║█████╗      ███████║██║  ██║██║   ██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║███████╗        
   ██║   ██╔══██║██╔══╝      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══╝  ██║╚██╗██║   ██║   ██║   ██║╚════██║        
   ██║   ██║  ██║███████╗    ██║  ██║██████╔╝ ╚████╔╝ ███████╗██║ ╚████║   ██║   ╚██████╔╝███████║        
   ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚══════╝        
 {white}                                                                                                         
 █████╗ ██╗    ████████╗███████╗███████╗████████╗    ███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗
██╔══██╗██║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ████╗ ████║██╔═══██╗██╔══██╗██║   ██║██║     ██╔════╝
███████║██║       ██║   █████╗  ███████╗   ██║       ██╔████╔██║██║   ██║██║  ██║██║   ██║██║     █████╗  
██╔══██║██║       ██║   ██╔══╝  ╚════██║   ██║       ██║╚██╔╝██║██║   ██║██║  ██║██║   ██║██║     ██╔══╝  
██║  ██║██║       ██║   ███████╗███████║   ██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝╚██████╔╝███████╗███████╗
╚═╝  ╚═╝╚═╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
 {green}                                                                                                         
 ██████╗ ██████╗ ████████╗    ██████╗    ███████╗                                                         
██╔════╝ ██╔══██╗╚══██╔══╝    ╚════██╗   ██╔════╝                                                         
██║  ███╗██████╔╝   ██║        █████╔╝   ███████╗                                                         
██║   ██║██╔═══╝    ██║        ╚═══██╗   ╚════██║                                                         
╚██████╔╝██║        ██║       ██████╔╝██╗███████║                                                         
 ╚═════╝ ╚═╝        ╚═╝       ╚═════╝ ╚═╝╚══════╝                                                         
"""

print(text)





print(f"{yellow}---------------------------------------------------------------------------------")
while True:
	query = input(f"Input 1 or 2\n1. Chatbot\n2. Update Vector Storage\n---------------------------------------------------------------------------------\n{white}$")
	if query == "exit" or query == "quit" or query == "q" or query == "f":
		print('Exiting')
		sys.exit()

	if query == '':
		continue

	if query == '1':
		break
	if query == '2':
		break
	else:
		input("Select 1 or 2")

if query == '1':
	import chatbot
elif query == '2':
	import vector_store
else:
	sys.exit()

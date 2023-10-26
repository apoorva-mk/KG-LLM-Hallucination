This doc outlines the procedure for SSHing into my desktop for 3090 GPU access.

First, some notes:

- This approach currently uses a service called [ngrok](https://ngrok.com/) to create a proxy server because my apartment's ISP doesn't allow for port forwarding.

### Syncthing
- The URL for this proxy server changes every time my computer gets restarted. So in order to maintain my connection, I use an open-source service called [syncthing](https://syncthing.net/) to store the current url for the ngrok server into a shared folder that gets synced to all of my devices, sort of like Google Drive. You are welcome to use this service aswell, and I can add you to the shared folder. Otherwise, you can just ping me for the latest url info.
  - To use this service, download the daemon associated with your OS, and start it up. There should be an option to "Show Device ID" or a QR Code. If you send that to me, I can add you to the device list for the shared folder.
  - After your device is added on my end, your daemon will detect that a folder called "Ngrok-creds" is available to download locally. You may need to specify the path to download to. This *should* be up-to-date with the latest url info, but lmk if you run into issues here.

### SSH Keys
- In order to set up SSH on your machine, follow the steps for setting it up on [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server). Once you get to the part about forwarding the public key to the server, instead you should send your public key file to me, and I can add it to the list of allowed hosts.
- Alternatively, if that approach runs into issues, ping me and I can send you the private key associated with my laptop.

### SSH via Command Line
- In order to ssh into the computer, use the following command:

```bash
ssh -p PORT zaristei@ADDRESS
```

- If you're getting this information from the ngrok creds folder, open the file you find there, and it will look something like this:

```json
{
  "name": "rdp-app",
  "public_url": "tcp://ADDRESS:PORT"
}
{
  "name": "sunshine",
  "public_url": "tcp://ADDRESS:PORT"
}
{
  "name": "ssh-app",
  "public_url": "tcp://ADDRESS:PORT"
}
```

- For SSH, use the information under the name "ssh-app", if you want to use windows RDP, use the info under rdp-app (this is password protected though). Ignore sunshine.

### SSH via VSCode (recommended)

- Once you make sure SSH works, I recommend downloading the SSH extension in your local VSCode, and interacting with the desktop while SSHing in within the IDE. You'll need to create/edit the profile you use for this every time the url changes.
- Once you SSH in, you probably won't be able to utilize the GPU on its own. In order to do that, I have a container running in the instance called "current_fedora". To access this container, you need to have the Dev Containers extension active on your VSCode. You can use this to "Connect to Existing Container", and that should open a new VSCode window that has full CUDA installed to it.

### Navigating the environment.
- WARNING!!!! This environment has full admin/root access to my Desktop system. Please Please Please don't try to install/uninstall anything on your own, and please constrain your edits to the repo location on my disk. I cloned the repo to `~/repos/KG-LLM-Hallucination`.
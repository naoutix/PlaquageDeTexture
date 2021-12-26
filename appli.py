def main():
    import tkinter
    from tkinter import filedialog
 
    #root = tkinter.Tk()
    dossier = filedialog.askopenfilename(title="Choisir la texture",filetypes=[("image",".png")],initialdir="./Textures/")
    if len(dossier)>0:
        print(dossier)
 
 
if __name__ == '__main__':
    main()
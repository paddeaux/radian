from tkinter import *
from tkinter import filedialog

from tkinter.ttk import *
from msc_pg import wrapper_function

window = Tk()

window.title("RADIAN - Synthetic Spatial Data Generation")

window.geometry('720x480')

# Row 1

filename_lbl = Label(window, text="Source Polygon (.geojson)", anchor=W)
filename_lbl.place(x=25, y=20)

filename_txt = Entry(window, width=45)
filename_txt.place(x=25, y=42)

def set_file_dir(text):
    filename_txt.delete(0, END)
    filename_txt.insert(0, text)
    return

def browse_files():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Geojson Polygons",
                                                        "*.geojson*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
    set_file_dir(filename)

filename_btn = Button(window, text="...", command=browse_files, width=5)
filename_btn.place(x = 302, y= 40)

totalpts_lbl = Label(window, text="Total Points")
totalpts_lbl.place(x=344, y=20)

totalpts_txt = Entry(window, width=25)
totalpts_txt.insert(0,500)
totalpts_txt.place(x=344, y=42)

ratio_lbl = Label(window, text="Ratio")
ratio_lbl.place(x=505, y=20)

ratio_txt = Entry(window, width=8)
ratio_txt.insert(0, 50)
ratio_txt.place(x=505, y=42)

gentype_lbl = Label(window, text="GenType")
gentype_lbl.place(x=560, y=20)

gentype_0 = Radiobutton(window,text='0', value=0).place(x = 565, y=42)
gentype_1 = Radiobutton(window,text='1', value=1).place(x = 600, y=42)
gentype_2 = Radiobutton(window,text='2', value=2).place(x = 635, y=42)
gentype_3 = Radiobutton(window,text='3', value=3).place(x = 665, y=42)

gentype = 0

if gentype_0:
    gentype = 0
elif gentype_1:
    gentype = 1
elif gentype_2:
    gentype = 2
elif gentype_3:
    gentype = 3

# Row 2

tablename_lbl = Label(window, text="SQL Table Name", anchor=W)
tablename_lbl.place(x=25, y=65)

tablename_txt = Entry(window, width=20)
tablename_txt.place(x=25, y=85)

intname_lbl = Label(window, text="Default Int Name", anchor=W)
intname_lbl.place(x=160, y=65)

intname_txt = Entry(window, width=15)
intname_txt.place(x=160, y=85)

strname_lbl = Label(window, text="Default String Name", anchor=W)
strname_lbl.place(x=160, y=110)

strname_txt = Entry(window, width=15)
strname_txt.place(x=160, y=130)

tsstart_lbl = Label(window, text="Timestamp Range", anchor=W)
tsstart_lbl.place(x=160, y=155)

tsstart_txt = Entry(window, width=15)
tsstart_txt.place(x=160, y=175)

tsstart_lbl = Label(window, text="Timestamp Range", anchor=W)
tsstart_lbl.place(x=160, y=155)

tsto_lbl = Label(window, text="to", anchor=W)
tsto_lbl.place(x=260, y=175)

tsend_txt = Entry(window, width=15)
tsend_txt.place(x=278, y=175)

vornum_lbl = Label(window, text="Secondary Voronoi Number")
vornum_lbl.place(x=560, y=65)

vornum_txt = Entry(window, width=5)
vornum_txt.insert(0, 8)
vornum_txt.place(x=560, y=85)

# Row 3

# Output checkboxes
to_sql = IntVar()
to_sql_chk = Checkbutton(window, text="Output to SQL", variable=to_sql)
to_sql_chk.place(x=560, y=110)

to_geojson = IntVar()
to_geojson_chk = Checkbutton(window, text="Output to GeoJSON", variable=to_geojson)
to_geojson_chk.place(x=560, y=130)

to_png = IntVar()
to_png_chk = Checkbutton(window, text="Output to PNG", variable=to_png)
to_png_chk.place(x=560, y=150)

to_plot = IntVar(value=1)
to_plot_chk = Checkbutton(window, text="Plot Output", variable=to_plot)
to_plot_chk.place(x=560, y=170)

to_breakdown = IntVar()
to_breakdown_chk = Checkbutton(window, text="Plot Breakdown", variable=to_breakdown)
to_breakdown_chk.place(x=560, y=190)

to_basemap = IntVar(value=1)
to_basemap_chk = Checkbutton(window, text="Plot Basemap (OSM)", variable=to_basemap)
to_basemap_chk.place(x=560, y=210)

# Generation Seed
var_list = ['to_plot']
    
def generate_seed():
    param_dict = {
        'total_pts' : totalpts_txt.get(),
        'ratio' : ratio_txt.get(),
        'gen_type' : gentype,
        'vor_num' : vornum_txt.get(),
        'to_sql' : to_sql.get(),
        'to_geojson' : to_geojson.get(),
        'to_png' : to_png.get(),
        'to_plot' : to_plot.get(),
        'to_breakdown' : to_breakdown.get(),
        'to_basemap' : to_basemap.get()
    }
    print(param_dict.values())

seed_lbl = Label(window, text="Generation Seed")
seed_lbl.place(x=228, y=250)

seed_txt = Entry(window, width=25)
seed_txt.place(x=320, y=250)

seed_btn = Button(window, text="Generate Seed", command=generate_seed)
seed_btn.place(x = 480, y= 248)

generate_btn = Button(window, text="Generate Points", command=wrapper_function)
generate_btn.place(x = 480, y= 320)

window.mainloop()
# What is DEVSimPy-plugin-activity-tracking
It is a plugin for DEVSimPy which track the activity of DEVSimPy atomic models.

#Depends
* [python-psutil](https://pypi.python.org/pypi/psutil) for cpu usage
* [networkx](https://networkx.github.io/) and [pylab](https://pypi.python.org/pypi/pylab) for graph
* [radon](https://pypi.python.org/pypi/radon) for metrics
* maccabe.py file
			 
#Installation
In order to view the blink plugin in the DEVSimPy plugin manager (Options->Preferences->Plugins), just:
* add the activity-tracking.py and the codepaths.py files into the "plugins" directory of DEVSimPy 
* add the string "activity-tracking" to the \__all\__ variable of the plugins/\__init\__.py file 

#Use
[video](https://youtu.be/HWG_Y22i8P8) to see the plugin in action. 

#Documentation

* Implementation and Analysis of DEVS Activity-Tracking with DEVSimPy, J.F. Santucci, L. Capocchi, ITM Web of Conferences 1 01001 (2013), DOI: 10.1051/itmconf/20130101001 [pdf]((http://www.itm-conferences.org/articles/itmconf/pdf/2013/01/itmconf_acti2012_01001.pdf)
* Using Activity Metrics for DEVS Simulation Profiling, A.  Muzy, L.Capocchi, J.F.Santucci, ITM Web of Conferences 3 01001 (2014), DOI: 10.1051/itmconf/20140301001 [pdf](http://www.itm-conferences.org/articles/itmconf/pdf/2014/02/itmconf_actims2014_01001.pdf)

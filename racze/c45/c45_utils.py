import math
from xml.etree import ElementTree as ET


def prettify(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            prettify(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

def isnum(attr):
    for x in set(attr):
        if not x=="?":
            try:
                x=float(x)
                return isinstance(x,float)
            except ValueError:
                return False
    return True

def entropy(x):
    ent=0
    for k in set(x):
        # x.count(k) adalah jumlah total data, kalau di excel berarti P & Q
        # len(x) adalah jumlah seluruh kasus, kalau di excel berarti O6
        p_i=float(x.count(k))/len(x)
        ent=ent - p_i* math.log(p_i,2)
    return ent

def chi(x, cat):
    ent=0.0
    # print 'START CRAMER--------------------------------------'
    # print 'Jumlah x ', len(x)
    # print 'x ', x
    # print 'Jumlah set x ', set(x)
    totalPerKelas = 0
    for k in set(x):
        totalPerKelas += x.count(k)
    for k in set(x):
        # print 'Jumlah K ', x.count(k)
        # print 'k ', k
        # print 'Jumlah kelas k ', k, ' total ', cat.count(k)
        # x.count(k) adalah jumlah total data, kalau di excel berarti P6
        # len(x) adalah jumlah seluruh kasus, kalau di excel berarti O6
        p_i = float(cat.count(k)) * float(len(x)) / float(len(cat))
        p_i = math.pow( float(x.count(k)) - float(p_i), 2) / float(x.count(k))
        # print 'cat.count(k) ', cat.count(k)
        # print 'len(x) ', len(x)
        # print 'len(cat) ', len(cat)
        # print 'p_i ', p_i
        ent = float(ent) + float(p_i)
    # print 'END CRAMER-------------------------------------- ', ent
    return ent

def cramer(countChase, chi_square):
    return math.sqrt(chi_square / (countChase * (2-1)))

def gain_ratio(category,attr,method):
    s=0
    chi_square=0.0
    cat=[]
    att=[]
    for i in range(len(attr)):
        if not attr[i]=="?":
            cat.append(category[i])
            att.append(attr[i])
    # print 'att ', att
    # print 'cat ', cat
    for i in set(att):
        #att.count(i)) = jumlah per atribut untuk semua kelas
        #len(att) = jumlah total kasus
        #p_i = jumlah per atribut untuk semua kelas / jumlah total kasus
        p_i=float(att.count(i))/len(att)
        cat_i=[]
        for j in range(len(cat)):
            if att[j]==i:
                cat_i.append(cat[j])
        s = s + p_i * entropy(cat_i)
        chi_square = float(chi_square) + float(chi(cat_i, cat))

    cramerValue = cramer(len(att), chi_square)
    # print chi_square, ' - ' , len(att) , ' - ', cramerValue, ' ________________________________________________________________________________________________________________________________________________'
    # s = penjumlahan entropy per value
    # gain adalah information gain
    # ent_att adalah split info
    # entropy(cat) adalah Entropy Total
    gain=entropy(cat)-s
    ent_att=entropy(att)
    if ent_att==0:
        return 0
    else:
        # return gain
        if method=='cramer':
            return gain/ent_att * cramerValue
            return math.pow( (gain/ent_att), cramerValue)
        else:
            return gain/ent_att


def gain(category, attr):
    cats=[]
    for i in range(len(attr)):
        if not attr[i]=="?":
            cats.append([float(attr[i]),category[i]])
    cats=sorted(cats, key=lambda x:x[0])
    
    cat=[cats[i][1] for i in range(len(cats))]
    att=[cats[i][0] for i in range(len(cats))]
    if len(set(att))==1:
        return 0
    else:
        gains=[]
        div_point=[]
        for i in range(1,len(cat)):
            if not att[i]==att[i-1]:
                gains.append(entropy(cat[:i])*float(i)/len(cat)+entropy(cat[i:])*(1-float(i)/len(cat)))
                div_point.append(i)
        gain=entropy(cat)-min(gains)
    
        p_1=float(div_point[gains.index(min(gains))])/len(cat)
        ent_attr= -p_1*math.log(p_1,2)-(1-p_1)*math.log((1-p_1),2)
        return gain/ent_attr

def division_point(category,attr):
    cats=[]
    for i in range(len(attr)):
        if not attr[i]=="?":
            cats.append([float(attr[i]),category[i]])
    cats=sorted(cats, key=lambda x:x[0])
    
    cat=[cats[i][1] for i in range(len(cats))]
    att=[cats[i][0] for i in range(len(cats))]
    gains=[]
    div_point=[]
    for i in range(1,len(cat)):
        if not att[i]==att[i-1]:
            gains.append(entropy(cat[:i])*float(i)/len(cat)+entropy(cat[i:])*(1-float(i)/len(cat)))
            div_point.append(i)
    return att[div_point[gains.index(min(gains))]]

def grow_tree(data,category,parent,attrs_names,method):
    if len(set(category))>1:
        
        division=[]
        for i in range(len(data)):
            if set(data[i])==set("?"):
                division.append(0)
            else:
                if (isnum(data[i])):
                    #field numerik
                    division.append(gain(category,data[i]))           
                else:
                    #field kategorikal
                    division.append(gain_ratio(category,data[i], method))
        if max(division)==0:
            num_max=0
            for cat in set(category):
                num_cat=category.count(cat)
                if num_cat>num_max:
                    num_max=num_cat
                    most_cat=cat                
            parent.text=most_cat
        else:
            index_selected=division.index(max(division))
            name_selected=str(attrs_names[index_selected])
            if isnum(data[index_selected]):
                div_point=division_point(category,data[index_selected])
                r_son_data=[[] for i in range(len(data))]
                r_son_category=[]
                l_son_data=[[] for i in range(len(data))]
                l_son_category=[]
                for i in range(len(category)):
                    if not data[index_selected][i]=="?":
                        if float(data[index_selected][i])<float(div_point):
                            l_son_category.append(category[i])
                            for j in range(len(data)):
                                l_son_data[j].append(data[j][i])     
                        else:
                            r_son_category.append(category[i])
                            for j in range(len(data)):
                                r_son_data[j].append(data[j][i])  
                if len(l_son_category)>0 and len(r_son_category)>0:
                    p_l=float(len(l_son_category))/(len(data[index_selected])-data[index_selected].count("?"))
                    son=ET.SubElement(parent,name_selected,{'value':str(div_point),"tanda":"kurang dari","flag":"l","p":str(round(p_l,3))})
                    grow_tree(l_son_data,l_son_category,son,attrs_names,method)
                    son=ET.SubElement(parent,name_selected,{'value':str(div_point),"tanda":"lebih dari sama dengan","flag":"r","p":str(round(1-p_l,3))})
                    grow_tree(r_son_data,r_son_category,son,attrs_names,method)
                else:
                    num_max=0
                    for cat in set(category):
                        num_cat=category.count(cat)
                        if num_cat>num_max:
                            num_max=num_cat
                            most_cat=cat                
                    parent.text=most_cat
            else:
                for k in set(data[index_selected]):
                    if not k=="?":
                        son_data=[[] for i in range(len(data))]
                        son_category=[]
                        for i in range(len(category)):
                            if data[index_selected][i]==k:
                                son_category.append(category[i])
                                for j in range(len(data)):
                                    son_data[j].append(data[j][i])
                        son=ET.SubElement(parent,name_selected,{'value':k,"flag":"m",'p':str(round(float(len(son_category))/(len(data[index_selected])-data[index_selected].count("?")),3))}) 
                        grow_tree(son_data,son_category,son,attrs_names,method)   
    else:
        parent.text=category[0]

def add(d1,d2):
    d=d1
    if d2 is None:
        d['unknown']=0
        return d
        # return d['unknown'] = 0
    else:
        for i in d2:
            if d.has_key(i):
                d[i]=d[i]+d2[i]
            else:
                d[i]=d2[i]
        return d

def decision(root,obs,attrs_names,p):
    if root.hasChildNodes():
        att_name=root.firstChild.nodeName
        if att_name=="#text":
            
            return decision(root.firstChild,obs,attrs_names,p)  
        else:
            att=obs[attrs_names.index(att_name)]
            if att=="?":
                d={}
                for child in root.childNodes:                    
                    d=add(d,decision(child,obs,attrs_names,p*float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag")=="m" and child.getAttribute("value")==att or \
                        child.getAttribute("flag")=="l" and float(att)<float(child.getAttribute("value")) or \
                        child.getAttribute("flag")=="r" and float(att)>=float(child.getAttribute("value")):
                        return decision(child,obs,attrs_names,p)    
    else:
        return {root.nodeValue:p}

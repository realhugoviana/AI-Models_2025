\LoadClass[a4paper,12pt]{article} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage[english]{babel} %Langue du document
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage[T1]{fontenc} %Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry} %Configuration de la page
\RequirePackage{ragged2e}


%-------------------- Informations sur le rapport ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\entreprise}[1]{\renewcommand{\entreprise}{#1}}
\newcommand{\tuteur}[1]{\renewcommand{\tuteur}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}
\newcommand{\myabstract}[1]{\renewcommand{\myabstract}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{
   % \includegraphics[scale=0.2]{logos/logo-im2ag-noir.png}
} %Affichage de l'image au top de la page
\chead{\begin{minipage}[b]{10cm} \centering \nouppercase{\leftmark} \end{minipage}}
\rhead{
   % \includegraphics[scale=0.2]{logos/logo-st-noir.png}
}
\rfoot{\thepage}
\cfoot{}
\lfoot{\titre}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}

    \centering %Centrer le contenu
        \includegraphics[scale=0.17]{logos/logo-couleur.png}
    \begin{flushleft}
        
    \end{flushleft}
    {\scshape EUR DS4H - Master 1 Informatique\par} %Nom de l'université
	{\scshape AI models and applications\par} %sous-titre
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\huge\bfseries \titre \par} \
    \rule{\linewidth}{0.2 mm} \\[0.5 cm]
	%\vspace{1cm}%Espace de 3cm

    %\begin{minipage}{0.5\textwidth}
        %\begin{flushleft} \large %Envoyer à gauche
            %\emph{\textbf{Students: }} %Pour le titre au dessus des noms à gauche
            {\Large \eleves} %Remplacer pour chacun
        %\end{flushleft}
	%\end{minipage}
    ~
    %\begin{minipage}{0.4\textwidth}
    %    \begin{flushleft} \large
    %        \emph{\textbf{Tutors :}} \\
    %         \enseignant \\
    %    \end{flushleft}
    %\end{minipage}
    \vspace{1cm}
    
    {\bfseries \large Abstract \par}
    \vspace{0.5cm}
    
    \justifying 
    {\noindent \myabstract \par}
    
    \centering
	\vfill
	{\large \the\year\par} %Affichage de la date

\end{titlepage}
}

\newcommand{\tabledematieres}{
\tableofcontents
\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}

\newcommand{\usection}[1]{\section*{#1}
\addcontentsline{toc}{section}{\protect\numberline{}#1}}



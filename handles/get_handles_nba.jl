# Scrape NBA Team Twitter Handles from a specific website

using Lazy

teams_url = "https://www.complex.com/sports/all-30-nba-twitter-accounts-ranked-for-2020/"
raw = read(download(teams_url), String)
# the @>> macro treats the value of the prior function as the *last* argument of the next function
teams = @>> begin
    findall(r"\(@\w+\)", raw)
    map(idx-> raw[idx])
    map(w-> strip(w,['(', ')', '@']))
    map(w-> lowercase(w))
    unique
    sort
    ws-> join(ws,'\n')
end

fd = open("handles_nba.csv","w")
print(fd,"handle\n")
print(fd,teams)
close(fd)

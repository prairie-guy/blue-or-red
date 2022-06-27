# Scrape Celebrity Twitter Handles

using Lazy

teams_url = "https://celebritycred.com/top-100-celebrity-twitter-accounts/"
raw = read(download(teams_url), String)
# the @>> macro treats the value of the prior function as the *last* argument of the next function
things = @>> begin
    findall(r"twitter\.com/\w+/?", raw)
    map(idx-> raw[idx])
    map(w->split(w,"/"))
    map(last)
    filter(!isempty)
    unique
    sort
    ws-> join(ws,'\n')
end

fd = open("handles_celebs.csv","w")
print(fd,"handle\n")
print(fd,things)
close(fd)
